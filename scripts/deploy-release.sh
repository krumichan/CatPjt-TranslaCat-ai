#!/usr/bin/env bash
# Run only after checking out the exact reviewed commit. Does not delete untracked model/dictionary files.
set -Eeuo pipefail
umask 077

SHA="${1:?Pass the exact release commit SHA}"
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "Invalid release SHA" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "$SHA" ]] || { echo "Checkout does not match release SHA" >&2; exit 2; }
: "${DEPLOY_ENV_B64:?Pass a base64 encoded environment file}"
IMAGE="translacat-ai:$SHA"
CONTAINER="fastapi-app"
# 운영에서 승인·검증한 인식 경로는 저장소 정책으로 유지한다.
# 비밀 환경 파일에서 누락되거나 오래된 값이 있어도 CPU 인식으로 되돌아가지 않는다.
VOICE_STT_PROVIDER="openai"
VOICE_REALTIME_PREVIEW_ENABLED="true"
BACKUP="${CONTAINER}-rollback-$(date -u +%Y%m%dT%H%M%S)-$$"
ENV_FILE="$(mktemp .deploy-env.XXXXXX)"
HAD_PREVIOUS=false
PREVIOUS_RUNNING=false
BACKED_UP=false
REPLACEMENT_ATTEMPTED=false

cleanup() { rm -f "$ENV_FILE"; }
trap cleanup EXIT

rollback() {
    local original_exit="$1"
    trap - ERR INT TERM
    set +e
    echo "Deployment failed (exit $original_exit); restoring the previous container." >&2
    if [ "$REPLACEMENT_ATTEMPTED" = true ]; then
        docker logs --tail 200 "$CONTAINER" >&2
        docker rm -f "$CONTAINER"
    fi
    if [ "$BACKED_UP" = true ]; then
        if docker rename "$BACKUP" "$CONTAINER"; then
            if [ "$PREVIOUS_RUNNING" = true ]; then
                docker start "$CONTAINER" || echo "CRITICAL: automatic rollback start failed" >&2
            fi
        else
            echo "CRITICAL: automatic rollback rename failed; preserved container: $BACKUP" >&2
        fi
    elif [ "$HAD_PREVIOUS" = true ] && [ "$PREVIOUS_RUNNING" = true ]; then
        # Stop may have succeeded just before rename failed.
        docker start "$CONTAINER" || echo "CRITICAL: could not restart existing container" >&2
    fi
    exit "$original_exit"
}
trap 'rollback $?' ERR
trap 'rollback 130' INT
trap 'rollback 143' TERM

printf '%s' "$DEPLOY_ENV_B64" | base64 --decode > "$ENV_FILE"
test -s "$ENV_FILE"
grep -Eq '^OPENAI_API_KEY=[^[:space:]]+' "$ENV_FILE" || {
    echo "Production Voice requires a configured OpenAI credential" >&2
    false
}
docker network inspect translacat-network >/dev/null 2>&1 || docker network create translacat-network
# All potentially destructive steps are after a successful immutable-tag build.
docker build --label "org.opencontainers.image.revision=$SHA" -t "$IMAGE" .

if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    PREVIOUS_RUNNING="$(docker inspect --format '{{.State.Running}}' "$CONTAINER")"
    if [ "$PREVIOUS_RUNNING" = true ]; then
        # 재생 중인 Voice가 있거나 상태 확인에 실패하면 중지 전에 배포를 실패시킨다.
        # 이전 이미지에도 최신 검사 코드를 stdin으로 전달하며 비밀 값은 컨테이너 안에서만 읽는다.
        docker exec -i "$CONTAINER" python - --require-idle < scripts/check_voice_release.py
    fi
    HAD_PREVIOUS=true
    docker stop --time 30 "$CONTAINER"
    docker rename "$CONTAINER" "$BACKUP"
    BACKED_UP=true
fi

REPLACEMENT_ATTEMPTED=true
docker run -d --name "$CONTAINER" --restart unless-stopped \
    --network translacat-network --env-file "$ENV_FILE" \
    --env "AI_VOICE_STT_PROVIDER=$VOICE_STT_PROVIDER" \
    --env "AI_VOICE_REALTIME_PREVIEW_ENABLED=$VOICE_REALTIME_PREVIEW_ENABLED" \
    -v translacat-ai-model-cache:/root/.cache/huggingface -p 8000:8000 "$IMAGE"

actual_revision="$(docker inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$CONTAINER")"
[[ "$actual_revision" == "$SHA" ]] || { echo "Running container revision mismatch" >&2; false; }

ATTEMPTS="${DEPLOY_HEALTH_ATTEMPTS:-60}"
DELAY="${DEPLOY_HEALTH_DELAY_SECONDS:-5}"
[[ "$ATTEMPTS" =~ ^[1-9][0-9]*$ && "$DELAY" =~ ^[0-9]+$ ]]
for ((attempt = 1; attempt <= ATTEMPTS; attempt++)); do
    status="$(docker inspect --format '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || true)"
    if [[ "$status" == "healthy" ]]; then
        # 일반 health는 CPU 인식도 정상으로 본다. 실제 모델까지 확인한 뒤 배포 성공으로 판정한다.
        docker exec "$CONTAINER" python scripts/check_voice_release.py
        echo "Released $SHA as $CONTAINER."
        if [ "$BACKED_UP" = true ]; then
            echo "Rollback container retained (stopped): $BACKUP"
        fi
        exit 0
    fi
    sleep "$DELAY"
done

echo "New container did not become ready" >&2
false # ERR trap rolls back and preserves a failing exit code for CI.
