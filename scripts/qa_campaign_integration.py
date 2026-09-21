"""Explicit, isolated local QA bootstrap; never connects to existing application DBs.

Preflight is read-only. Prepare writes only a newly owned private directory.
Start-infra is a separate explicit action: new labelled Docker network/volumes and
containers only, with pinned cached images and loopback ports. Separate start-be
and check-auth actions launch the isolated server and normal auth reproduction.
No cleanup/delete, provider calls, or production config loading is included.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request
import zipfile


OWNER = "translacat-isolated-qa"
DOCKER = Path("C:/Program Files/Docker/Docker/resources/bin/docker.exe")
_QA_PORT_SHIFT = int(os.environ.get("TRANSLACAT_QA_PORT_SHIFT", "0"))
if _QA_PORT_SHIFT not in {0, 2}:
    raise ValueError("Unsupported isolated QA port profile")
PORTS = {"mysql": 13316 + _QA_PORT_SHIFT, "redis": 16389 + _QA_PORT_SHIFT,
         "be": 18081 + _QA_PORT_SHIFT, "ai": 18082 + _QA_PORT_SHIFT,
         "fe": 13010}
IMAGE_REFS = {"mysql": "mysql:latest", "redis": "redis:7.4.10-alpine"}
QA_LOGBACK = """<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <appender name="QA_CONSOLE" class="ch.qos.logback.core.ConsoleAppender">
    <encoder><pattern>%date %-5level %logger{40} - %msg%n</pattern></encoder>
  </appender>
  <logger name="jp.co.translacat.global.logging.ApiLoggingFilter" level="OFF"/>
  <logger name="jdbc" level="OFF"/>
  <root level="INFO"><appender-ref ref="QA_CONSOLE"/></root>
</configuration>
"""


def _docker(*args: str, environment: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        [str(DOCKER), *args], env=environment, capture_output=True,
        text=True, timeout=45, check=False,
    )
    if result.returncode:
        # Do not echo Docker config, environment, command output, or secrets.
        raise RuntimeError(f"Docker operation failed: {args[0]}, exit={result.returncode}")
    return result.stdout.strip()


def _campaign(value: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9-]{2,39}", value):
        raise ValueError("campaign must be 3..40 lowercase letters/digits/hyphens")
    return value


def resource_names(campaign: str) -> dict[str, str]:
    prefix = f"{OWNER}-{_campaign(campaign)}"
    return {
        "network": f"{prefix}-net",
        "mysqlContainer": f"{prefix}-mysql",
        "redisContainer": f"{prefix}-redis",
        "mysqlVolume": f"{prefix}-mysql-data",
        "redisVolume": f"{prefix}-redis-data",
        "database": f"translacat_qa_{campaign.replace('-', '_')}",
    }


def _ports_free() -> dict[str, bool]:
    result = {}
    for name, port in PORTS.items():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                result[name] = False
            else:
                result[name] = True
    return result


def _published_ports_ready() -> dict[str, bool]:
    result = {}
    for kind in ("mysql", "redis"):
        try:
            with socket.create_connection(("127.0.0.1", PORTS[kind]), timeout=2):
                result[kind] = True
        except OSError:
            result[kind] = False
    return result


def _verify_local_docker() -> None:
    if os.environ.get("DOCKER_HOST"):
        raise ValueError("DOCKER_HOST override is forbidden for isolated host QA")
    context = _docker("context", "show")
    endpoint = _docker("context", "inspect", context, "--format", "{{.Endpoints.docker.Host}}")
    if endpoint not in {"npipe:////./pipe/docker_engine", "npipe:////./pipe/dockerDesktopLinuxEngine"}:
        raise ValueError("Docker daemon is not a verified local Windows named pipe")


def preflight(campaign: str) -> dict[str, Any]:
    _verify_local_docker()
    names = resource_names(campaign)
    existing = {
        "containers": set(_docker("ps", "-a", "--format", "{{.Names}}").splitlines()),
        "volumes": set(_docker("volume", "ls", "--format", "{{.Name}}").splitlines()),
        "networks": set(_docker("network", "ls", "--format", "{{.Name}}").splitlines()),
    }
    collisions = [
        name for key, name in names.items()
        if key != "database" and name in set().union(*existing.values())
    ]
    images: dict[str, Any] = {}
    for name, reference in IMAGE_REFS.items():
        identity = json.loads(_docker(
            "image", "inspect", reference,
            "--format", '{"id":{{json .Id}},"digests":{{json .RepoDigests}}}',
        ))
        images[name] = identity
    ports = _ports_free()
    return {
        "owner": OWNER, "campaign": campaign, "names": names,
        "ports": PORTS, "portsFree": ports,
        "resourceCollisions": collisions, "images": images,
        "readyForFreshInfrastructure": all(ports.values()) and not collisions,
        "databaseBoundary": "NEW_CONTAINER_NEW_VOLUME_ONLY_NO_EXISTING_DB_CONNECTION",
        "feAuthentication": "Google OAuth only; no synthetic cookie/JWT bypass",
        "applicationLaunch": "NOT_STARTED",
        "hostNetworkBoundary": "Container network is internal; host application egress needs separate control",
    }


def _write_json(path: Path, value: Any) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        # Windows scanners/readers may briefly hold this diagnostic file open.
        # Retain the new complete snapshot and retry only sharing/access races;
        # never replace the artifact with a partially written JSON document.
        for attempt in range(4):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(0.02 * (attempt + 1))
    finally:
        Path(temporary).unlink(missing_ok=True)


def be_properties(directory: Path, names: dict[str, str]) -> str:
    # Load this with classpath:/application.properties + this exact file, profile qa.
    # Never activate local/prod, whose storage/data source values are not QA-safe.
    properties = {
        "logging.config": (directory / "logback-qa.xml").as_uri(),
        "server.address": "127.0.0.1",
        "server.port": str(PORTS["be"]),
        "spring.datasource.driver-class-name": "com.mysql.cj.jdbc.Driver",
        "spring.datasource.url": (
            f"jdbc:mysql://127.0.0.1:{PORTS['mysql']}/{names['database']}"
            "?useSSL=false&allowPublicKeyRetrieval=true&serverTimezone=UTC"
        ),
        "spring.datasource.username": "qa_app",
        "spring.datasource.password": "${QA_DB_PASSWORD}",
        "spring.jpa.hibernate.ddl-auto": "update",
        "spring.sql.init.mode": "never",
        "spring.data.redis.host": "127.0.0.1",
        "spring.data.redis.port": str(PORTS["redis"]),
        "translacat.redis.verify-on-startup": "true",
        "ai-server.url": f"http://127.0.0.1:{PORTS['ai']}",
        "ai-server.api-key": "${QA_AI_API_KEY}",
        "jwt.token.secret-key": "${QA_JWT_SECRET}",
        "jwt.token.expired.access": "3600000",
        "jwt.token.expired.refresh": "86400000",
        "cors.allowed-origin": (
            f"http://127.0.0.1:{PORTS['fe']},http://localhost:{PORTS['fe']},"
            "http://127.0.0.1:3000,http://localhost:3000"
        ),
        "spring.security.oauth2.client.registration.google.client-id": "${QA_GOOGLE_CLIENT_ID:qa-disabled}",
        "spring.ai.google.genai.api-key": "qa-disabled-not-a-real-key",
        "external.google.proxy-url": "http://127.0.0.1:9/qa-disabled",
        "external.use-proxy": "true",
        "sudachi.dictionary.path": "src/main/resources/system_full.dic",
        "translacat.storage.type": "local",
        "translacat.storage.local.root-path": (directory / "storage").as_posix(),
        "translacat.storage.local.public-base-url": f"http://127.0.0.1:{PORTS['be']}/api/v1/public/storage",
        "translacat.ai.chat-translation.mode": "temporary",
        "translacat.chat.presence.enabled": "false",
        "translacat.batch.chat-translation-retry.enabled": "false",
        "translacat.batch.ai-revival.enabled": "false",
        "translacat.batch.fixed-cost.enabled": "false",
        "translacat.voice.enabled": "false",
        # Defense in depth: the new DB also defaults pool replenishment OFF.
        "language-learning.level-test.question-pool.replenish-initial-delay-ms": "86400000",
        "logging.level.jp.co.translacat.global.logging.ApiLoggingFilter": "OFF",
        "logging.level.jdbc": "OFF",
        "logging.level.org.hibernate.SQL": "OFF",
        "logging.level.org.hibernate.orm.jdbc.bind": "OFF",
    }
    return "\n".join(f"{key}={value}" for key, value in properties.items()) + "\n"


def prepare(directory: Path, campaign: str) -> dict[str, Any]:
    report = preflight(campaign)
    if not report["readyForFreshInfrastructure"]:
        raise ValueError("QA resource/port collision; no existing resources may be reused")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    if os.name == "nt":
        principal = f"{os.environ['USERDOMAIN']}\\{os.environ['USERNAME']}"
        secured = subprocess.run(
            ["icacls", str(directory), "/inheritance:r", "/grant:r", f"{principal}:(OI)(CI)F"],
            capture_output=True, check=False, timeout=10,
        )
        if secured.returncode:
            raise RuntimeError("Private artifact ACL setup failed; no secrets written")
    else:
        directory.chmod(0o700)
    secret_values = {
        "QA_DB_PASSWORD": secrets.token_urlsafe(32),
        "QA_DB_ROOT_PASSWORD": secrets.token_urlsafe(32),
        "QA_AI_API_KEY": secrets.token_urlsafe(32),
        "QA_JWT_SECRET": base64.b64encode(secrets.token_bytes(64)).decode("ascii"),
    }
    secrets_file = directory / "secrets.private.json"
    _write_json(secrets_file, secret_values)
    if os.name != "nt":
        secrets_file.chmod(0o600)
    (directory / "application-qa.properties").write_text(
        be_properties(directory, report["names"]), encoding="utf-8",
    )
    (directory / "logback-qa.xml").write_text(QA_LOGBACK, encoding="utf-8")
    report.update({"status": "PREPARED", "directory": str(directory), "createdResources": []})
    _write_json(directory / "manifest.json", report)
    return report


def start_infrastructure(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    campaign = _campaign(manifest["campaign"])
    if manifest.get("owner") != OWNER or manifest.get("status") != "PREPARED":
        raise ValueError("Only an unstarted owned QA manifest may create infrastructure")
    current = preflight(campaign)
    if not current["readyForFreshInfrastructure"]:
        raise ValueError("QA resource or port collision; start refused")
    if current["images"] != manifest["images"]:
        raise ValueError("Cached image identities changed since preparation")
    names = resource_names(campaign)
    if manifest["names"] != names or manifest["ports"] != PORTS:
        raise ValueError("Manifest target mismatch")
    secret_values = json.loads((directory / "secrets.private.json").read_text(encoding="utf-8"))
    labels = ["--label", f"qa.owner={OWNER}", "--label", f"qa.campaign={campaign}"]
    created = manifest["createdResources"]
    try:
        network_id = _docker("network", "create", "--internal", *labels, names["network"])
        created.append({"type": "network", "name": names["network"], "id": network_id})
        _write_json(directory / "manifest.json", manifest)
        for resource in ("mysqlVolume", "redisVolume"):
            _docker("volume", "create", *labels, names[resource])
            created.append({"type": "volume", "name": names[resource]})
            _write_json(directory / "manifest.json", manifest)
        for kind, internal_port, mount in (("mysql", 3306, "/var/lib/mysql"), ("redis", 6379, "/data")):
            environment = os.environ.copy()
            environment.update({
                "MYSQL_DATABASE": names["database"], "MYSQL_USER": "qa_app",
                "MYSQL_PASSWORD": secret_values["QA_DB_PASSWORD"],
                "MYSQL_ROOT_PASSWORD": secret_values["QA_DB_ROOT_PASSWORD"],
            })
            args = [
                "run", "--detach", "--pull=never", "--restart=no", *labels,
                "--name", names[f"{kind}Container"], "--network", names["network"],
                "--publish", f"127.0.0.1:{PORTS[kind]}:{internal_port}",
                "--volume", f"{names[f'{kind}Volume']}:{mount}",
            ]
            if kind == "mysql":
                for key in ("MYSQL_DATABASE", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_ROOT_PASSWORD"):
                    args.extend(("--env", key))
            args.append(manifest["images"][kind]["id"])
            container_id = _docker(*args, environment=environment)
            created.append({"type": "container", "name": names[f"{kind}Container"], "id": container_id})
            _write_json(directory / "manifest.json", manifest)
        manifest["status"] = "INFRASTRUCTURE_STARTED_NOT_APPLICATION_READY"
    except BaseException:
        manifest["status"] = "PARTIAL_INFRASTRUCTURE_FAILURE_NO_AUTOMATIC_DELETION"
        raise
    finally:
        _write_json(directory / "manifest.json", manifest)
    return manifest


def _owned_manifest(directory: Path) -> dict[str, Any]:
    _verify_local_docker()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("owner") != OWNER or manifest.get("names") != resource_names(manifest["campaign"]):
        raise ValueError("Not an owned QA manifest")
    if manifest.get("ports") != PORTS:
        raise ValueError("QA port manifest changed")
    for kind in ("mysql", "redis"):
        name = manifest["names"][f"{kind}Container"]
        runtime = json.loads(_docker(
            "inspect", name, "--format",
            '{"id":{{json .Id}},"labels":{{json .Config.Labels}},"ports":{{json .HostConfig.PortBindings}}}',
        ))
        recorded = next(item for item in manifest["createdResources"] if item.get("name") == name)
        if runtime["id"] != recorded["id"] or runtime["labels"].get("qa.owner") != OWNER:
            raise ValueError("QA container identity/ownership mismatch")
        if runtime["labels"].get("qa.campaign") != manifest["campaign"]:
            raise ValueError("QA campaign ownership mismatch")
        internal_port = "3306/tcp" if kind == "mysql" else "6379/tcp"
        if runtime["ports"].get(internal_port) != [{"HostIp": "127.0.0.1", "HostPort": str(PORTS[kind])}]:
            raise ValueError("QA container binding mismatch")
    return manifest


def check_infrastructure(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    manifest = _owned_manifest(directory)
    private = json.loads((directory / "secrets.private.json").read_text(encoding="utf-8"))
    environment = os.environ.copy()
    environment["MYSQL_PWD"] = private["QA_DB_PASSWORD"]
    database = manifest["names"]["database"]
    result = _docker(
        "exec", "--env", "MYSQL_PWD", manifest["names"]["mysqlContainer"],
        "mysql", "--user=qa_app", "--batch", "--skip-column-names", database,
        "--execute=SELECT DATABASE(), VERSION();", environment=environment,
    )
    selected_database, version = result.split("\t")
    if selected_database != database:
        raise ValueError("Connected database did not match newly owned QA database")
    redis = _docker("exec", manifest["names"]["redisContainer"], "redis-cli", "ping")
    if redis != "PONG":
        raise ValueError("QA Redis readiness failed")
    manifest["infraHealth"] = {
        "mysqlDatabase": selected_database, "mysqlVersion": version,
        "redis": redis, "hostPublishedPortsReady": _published_ports_ready(),
    }
    manifest["hostNetworkBoundary"] = (
        "Dedicated labelled QA instances, schemas and loopback ports; outbound network isolation NOT guaranteed"
    )
    if manifest.get("hostNetwork"):
        manifest["hostNetwork"]["hostPublishedPortsReady"] = manifest["infraHealth"]["hostPublishedPortsReady"]
    _write_json(directory / "manifest.json", manifest)
    return manifest


def attach_host_network(directory: Path) -> dict[str, Any]:
    """Allow host-published QA ports without enabling outbound masquerading."""
    directory = directory.resolve()
    manifest = _owned_manifest(directory)
    if manifest.get("hostNetwork"):
        raise ValueError("Host QA network already recorded; inspect instead of duplicating")
    name = f"{manifest['names']['network']}-host"
    existing = _docker("network", "ls", "--format", "{{.Name}}").splitlines()
    if name in existing:
        raise ValueError("Host QA network name collision")
    identity = _docker(
        "network", "create", "--driver", "bridge",
        "--opt", "com.docker.network.bridge.enable_ip_masquerade=false",
        "--label", f"qa.owner={OWNER}", "--label", f"qa.campaign={manifest['campaign']}", name,
    )
    manifest["hostNetwork"] = {"name": name, "id": identity, "masquerade": False, "connectedContainers": []}
    _write_json(directory / "manifest.json", manifest)
    for kind in ("mysql", "redis"):
        container = manifest["names"][f"{kind}Container"]
        _docker("network", "connect", name, container)
        manifest["hostNetwork"]["connectedContainers"].append(container)
        _write_json(directory / "manifest.json", manifest)
    options = json.loads(_docker("network", "inspect", name, "--format", "{{json .Options}}"))
    if options.get("com.docker.network.bridge.enable_ip_masquerade") != "false":
        raise ValueError("QA host network unexpectedly enables outbound masquerading")
    manifest["hostNetwork"]["verifiedOptions"] = options
    manifest["hostNetwork"]["hostPublishedPortsReady"] = _published_ports_ready()
    _write_json(directory / "manifest.json", manifest)
    return manifest


def start_be(directory: Path, be_repository: Path, java: Path) -> dict[str, Any]:
    directory = directory.resolve()
    manifest = check_infrastructure(directory)
    if not all(manifest["infraHealth"]["hostPublishedPortsReady"].values()):
        raise ValueError("Container-internal readiness is not host-published readiness; BE launch refused")
    if not _ports_free()["be"]:
        raise ValueError("QA BE port occupied; refusing duplicate launch")
    previous = manifest.get("beProcess")
    if previous:
        if os.name != "nt":
            raise ValueError("Existing process record requires explicit host verification")
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {int(previous['pid'])}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        if any(len(row) > 1 and row[1] == str(previous["pid"]) for row in csv.reader(result.stdout.splitlines())):
            raise ValueError("Recorded BE process still exists; refusing duplicate launch")
        manifest.setdefault("bePreviousStarts", []).append(previous)
    if (directory / "application-qa.properties").read_text(encoding="utf-8") != be_properties(directory, manifest["names"]):
        raise ValueError("QA properties changed; refusing unverified configuration")
    jars = [path for path in (be_repository / "build" / "libs").glob("*.jar") if not path.name.endswith("-plain.jar")]
    if len(jars) != 1:
        raise ValueError("Expected exactly one current BE bootJar")
    private = json.loads((directory / "secrets.private.json").read_text(encoding="utf-8"))
    # Inherited Spring/JVM options can override file configuration and redirect
    # traffic to a developer DB. Preserve only platform/runtime paths, then add
    # fresh isolated credentials; never inherit application/provider settings.
    platform_keys = {
        "SYSTEMROOT", "WINDIR", "PATH", "JAVA_HOME", "TEMP", "TMP",
        "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA",
        "COMSPEC", "PROGRAMDATA", "USERDOMAIN", "USERNAME",
    }
    environment = {key: value for key, value in os.environ.items() if key.upper() in platform_keys}
    environment.update(private)
    # Explicit config/profile arguments override any developer-local active profile.
    command = [
        str(java.resolve()), "-jar", str(jars[0].resolve()),
        "--spring.profiles.active=qa",
        f"--spring.config.location=classpath:/application.properties,file:{(directory / 'application-qa.properties').as_posix()}",
    ]
    with (directory / "be.stdout.log").open("ab") as output, (directory / "be.stderr.log").open("ab") as error:
        process = subprocess.Popen(
            command, cwd=be_repository, env=environment, stdin=subprocess.DEVNULL,
            stdout=output, stderr=error,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    manifest["beProcess"] = {"pid": process.pid, "command": command, "cwd": str(be_repository.resolve())}
    manifest["applicationLaunch"] = "BE_STARTED_HEALTH_NOT_YET_VERIFIED"
    _write_json(directory / "manifest.json", manifest)
    return manifest


def refresh_qa_config(directory: Path) -> dict[str, Any]:
    """Refresh only this tool's owned QA config, never source/local/prod files."""
    directory = directory.resolve()
    manifest = _owned_manifest(directory)
    if not _ports_free()["be"]:
        raise ValueError("Do not rewrite config of a running QA BE")
    (directory / "application-qa.properties").write_text(be_properties(directory, manifest["names"]), encoding="utf-8")
    (directory / "logback-qa.xml").write_text(QA_LOGBACK, encoding="utf-8")
    return {"status": "OWNED_QA_CONFIG_REFRESHED", "campaign": manifest["campaign"]}


def configure_qa_google(directory: Path, fe_repository: Path, node: Path) -> dict[str, Any]:
    """Copy only the normal FE OAuth audience to private QA env, never credentials/stdout.

    Use the FE's own @next/env development precedence, matching its QA launcher.
    This enables real Google token validation, not synthetic authentication.
    """
    directory = directory.resolve()
    manifest = _owned_manifest(directory)
    if not (fe_repository / "node_modules" / "@next" / "env").is_dir():
        raise ValueError("Existing FE @next/env runtime required; no dependency installation")
    code = (
        "const root=process.argv[1];"
        "const env=require(require.resolve('@next/env',{paths:[root]}));"
        "env.loadEnvConfig(root,true,{info(){},error(){}});"
        "process.stdout.write(JSON.stringify({clientId:process.env.GOOGLE_CLIENT_ID}));"
    )
    result = subprocess.run(
        [str(node), "-e", code, str(fe_repository.resolve())],
        cwd=fe_repository, capture_output=True, text=True, timeout=20,
        check=False, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode:
        raise ValueError("FE OAuth audience resolution failed; output suppressed")
    try:
        client_id = json.loads(result.stdout).get("clientId")
    except (ValueError, AttributeError):
        raise ValueError("Invalid FE OAuth audience resolution; output suppressed") from None
    if not isinstance(client_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+\.apps\.googleusercontent\.com", client_id):
        raise ValueError("FE Google OAuth client ID absent or invalid; output suppressed")
    private_path = directory / "secrets.private.json"
    private = json.loads(private_path.read_text(encoding="utf-8"))
    private["QA_GOOGLE_CLIENT_ID"] = client_id
    _write_json(private_path, private)
    report = {
        "status": "QA_GOOGLE_AUDIENCE_PREPARED_RESTART_REQUIRED",
        "campaign": manifest["campaign"],
        "clientIdSha256": hashlib.sha256(client_id.encode("utf-8")).hexdigest(),
        "resolution": "FE @next/env.loadEnvConfig(repository, true); normal Google validation",
        "productionConfigurationChanged": False,
    }
    _write_json(directory / "google-audience-configuration.json", report)
    return report


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Redirect outside the exact QA API target is forbidden")


def _qa_request(path: str, payload: dict[str, str] | None = None) -> tuple[int, dict[str, Any]]:
    if path not in {"/api/v1/health", "/api/v1/auth/register", "/api/v1/auth/login"}:
        raise ValueError("Bootstrap may not trigger learning generation or arbitrary API endpoints")
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORTS['be']}{path}", data=data,
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        response = opener.open(request, timeout=20)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        body = response.read()
        try:
            parsed = json.loads(body)
        except (ValueError, UnicodeError):
            parsed = {}
        return response.code, parsed if isinstance(parsed, dict) else {}


def check_auth(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    manifest = _owned_manifest(directory)
    if not manifest.get("beProcess"):
        raise ValueError("No recorded QA BE process")
    artifact = directory / "auth-reproduction.json"
    if artifact.exists():
        raise ValueError("Auth reproduction is one-shot; existing evidence must be preserved")
    health, _ = _qa_request("/api/v1/health")
    if health != 200:
        raise ValueError("QA BE not healthy")
    secret_file = directory / "secrets.private.json"
    private = json.loads(secret_file.read_text(encoding="utf-8"))
    private["QA_APP_EMAIL"] = f"qa-{manifest['campaign']}@example.invalid"
    private["QA_APP_PASSWORD"] = secrets.token_urlsafe(24)
    _write_json(secret_file, private)
    register_status, registered = _qa_request("/api/v1/auth/register", {
        "email": private["QA_APP_EMAIL"], "password": private["QA_APP_PASSWORD"],
        "username": "Isolated QA",
    })
    login_status, logged_in = _qa_request("/api/v1/auth/login", {
        "email": private["QA_APP_EMAIL"], "password": private["QA_APP_PASSWORD"],
    })
    login_body = logged_in.get("body") or {}
    token = login_body.get("accessToken") if isinstance(login_body, dict) else None
    if isinstance(token, str):
        private["QA_APP_ACCESS_TOKEN"] = token
        _write_json(secret_file, private)
    registered_body = registered.get("body") or {}
    report = {
        "campaign": manifest["campaign"], "healthStatus": health,
        "registerHttpStatus": register_status, "loginHttpStatus": login_status,
        "registeredUserId": registered_body.get("id") if isinstance(registered_body, dict) else None,
        "accessTokenReceived": isinstance(token, str),
        "providerCalls": 0,
        "boundary": "Only owned QA health/register/login; no authentication bypass",
    }
    _write_json(artifact, report)
    return report


def _bcrypt_fixture_hash(directory: Path, boot_jar: Path, java: Path, password: str) -> str:
    """Use the built app's BCrypt implementation; never expose a credential in argv."""
    fixture_dir = directory / "auth-fixture"
    fixture_dir.mkdir(exist_ok=False)
    with zipfile.ZipFile(boot_jar) as archive:
        members = [name for name in archive.namelist() if re.fullmatch(
            r"BOOT-INF/lib/spring-security-crypto-[^/]+\.jar", name,
        )]
        if len(members) != 1:
            raise ValueError("Expected built app's single security crypto dependency")
        dependency = fixture_dir / "spring-security-crypto.jar"
        dependency.write_bytes(archive.read(members[0]))
    source = fixture_dir / "QaBcryptFixture.java"
    source.write_text(
        "import org.springframework.security.crypto.bcrypt.BCrypt;\n"
        "class QaBcryptFixture { public static void main(String[] args) throws Exception {\n"
        "String password = new String(System.in.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);\n"
        "System.out.print(BCrypt.hashpw(password, BCrypt.gensalt(12)));\n"
        "}}\n", encoding="utf-8",
    )
    environment = {key: value for key, value in os.environ.items() if key.upper() in {
        "SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
    }}
    result = subprocess.run(
        [str(java), "--class-path", str(dependency), str(source)], input=password,
        capture_output=True, text=True, timeout=30, check=False, env=environment,
    )
    encoded = result.stdout.strip()
    if result.returncode or not re.fullmatch(r"\$2[aby]\$12\$[./A-Za-z0-9]{53}", encoded):
        raise ValueError("Private BCrypt fixture generation failed")
    return encoded


def _qa_mysql_query(manifest: dict[str, Any], private: dict[str, str], sql: str) -> str:
    environment = os.environ.copy()
    environment["MYSQL_PWD"] = private["QA_DB_PASSWORD"]
    result = subprocess.run([
        str(DOCKER), "exec", "--interactive", "--env", "MYSQL_PWD",
        manifest["names"]["mysqlContainer"], "mysql", "--user=qa_app",
        "--batch", "--skip-column-names", manifest["names"]["database"],
    ], input=sql, capture_output=True, text=True, timeout=20, check=False, env=environment)
    if result.returncode:
        raise RuntimeError("Owned QA fixture SQL failed; secret output suppressed")
    return result.stdout.strip()


def seed_qa_auth_fixture(directory: Path, java: Path) -> dict[str, Any]:
    """Explicit repair of this run's synthetic QA fixture, not an auth production fix."""
    directory = directory.resolve()
    manifest = _owned_manifest(directory)
    original = json.loads((directory / "auth-reproduction.json").read_text(encoding="utf-8"))
    artifact = directory / "auth-fixture-result.json"
    if artifact.exists():
        raise ValueError("Auth fixture seed is one-shot; preserve prior evidence")
    identity = original.get("registeredUserId")
    if (original.get("campaign") != manifest["campaign"] or type(identity) is not int
            or identity <= 0 or original.get("registerHttpStatus") != 200
            or original.get("accessTokenReceived") is not False):
        raise ValueError("No exact failed synthetic registration is eligible for fixture seed")
    private_path = directory / "secrets.private.json"
    private = json.loads(private_path.read_text(encoding="utf-8"))
    if private["QA_APP_EMAIL"] != f"qa-{manifest['campaign']}@example.invalid":
        raise ValueError("Fixture must be this campaign's synthetic account")
    email_hex = private["QA_APP_EMAIL"].encode("utf-8").hex()
    password_hex = private["QA_APP_PASSWORD"].encode("utf-8").hex()
    predicate = (
        f"id={identity} AND email=CONVERT(0x{email_hex} USING utf8mb4) "
        f"AND password=CONVERT(0x{password_hex} USING utf8mb4) AND social_type='LOCAL'"
    )
    if _qa_mysql_query(manifest, private, f"SELECT COUNT(*) FROM user WHERE {predicate};") != "1":
        raise ValueError("Exact newly registered raw-password fixture not found; no write")
    command = manifest["beProcess"]["command"]
    boot_jar = Path(command[command.index("-jar") + 1])
    encoded = _bcrypt_fixture_hash(directory, boot_jar, java, private["QA_APP_PASSWORD"])
    report = {
        "campaign": manifest["campaign"], "registeredUserId": identity,
        "status": "FIXTURE_PREPARED", "encoding": "app BCrypt cost12",
        "providerCalls": 0, "productionSourceChanged": False,
        "boundary": "Only exact newly registered synthetic user; normal auth login remains required",
    }
    _write_json(artifact, report)
    encoded_hex = encoded.encode("utf-8").hex()
    updated = _qa_mysql_query(manifest, private, (
        f"UPDATE user SET password=CONVERT(0x{encoded_hex} USING utf8mb4) WHERE {predicate}; SELECT ROW_COUNT();"
    ))
    if updated != "1":
        raise ValueError("QA fixture seed did not affect exactly one expected row")
    report["status"] = "FIXTURE_UPDATED_LOGIN_PENDING"
    _write_json(artifact, report)
    login_status, body = _qa_request("/api/v1/auth/login", {
        "email": private["QA_APP_EMAIL"], "password": private["QA_APP_PASSWORD"],
    })
    token = (body.get("body") or {}).get("accessToken")
    if isinstance(token, str) and token:
        private["QA_APP_ACCESS_TOKEN"] = token
        _write_json(private_path, private)
    report.update({"status": "COMPLETED", "loginHttpStatus": login_status, "accessTokenReceived": bool(token)})
    _write_json(artifact, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "prepare", "start-infra", "check-infra", "attach-host-network", "start-be", "check-auth", "refresh-config", "seed-qa-auth-fixture", "configure-qa-google"))
    parser.add_argument("--campaign", default="astra-20260919")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--be-repository", type=Path, default=Path("C:/workspace/project-cat/CatPjt-TranslaCat-be"))
    parser.add_argument("--java", type=Path, default=Path("C:/Users/lovel/.jdks/corretto-21.0.3/bin/java.exe"))
    parser.add_argument("--fe-repository", type=Path, default=Path("C:/workspace/project-cat/CatPjt-TranslaCat-fe"))
    parser.add_argument("--node", type=Path, default=Path("C:/nvm4w/nodejs/node.exe"))
    args = parser.parse_args()
    if args.action == "preflight":
        report = preflight(args.campaign)
    else:
        if args.directory is None:
            parser.error("--directory is required for explicit write/start actions")
        if args.action == "prepare":
            report = prepare(args.directory, args.campaign)
        elif args.action == "start-infra":
            report = start_infrastructure(args.directory)
        elif args.action == "check-infra":
            report = check_infrastructure(args.directory)
        elif args.action == "attach-host-network":
            report = attach_host_network(args.directory)
        elif args.action == "start-be":
            report = start_be(args.directory, args.be_repository, args.java)
        elif args.action == "refresh-config":
            report = refresh_qa_config(args.directory)
        elif args.action == "seed-qa-auth-fixture":
            report = seed_qa_auth_fixture(args.directory, args.java)
        elif args.action == "configure-qa-google":
            report = configure_qa_google(args.directory, args.fe_repository, args.node)
        else:
            report = check_auth(args.directory)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
