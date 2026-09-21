"""Offline receipt currency validation; independent of enabled account-book currencies.

SIX ISO 4217 List One snapshot, retrieved 2026-09-19:
https://www.six-group.com/dam/download/financial-information/data-center/iso-currrency/lists/list-one.xml
Includes numeric minor units; excludes nonmonetary/test/no-currency codes.
BGN remains accepted for historic receipts (List Three effective 2026-01-01).
Refresh this catalog when the maintenance agency publishes new currency codes.
"""

_TWO_DECIMAL = """
AED AFN ALL AMD AOA ARS AUD AWG AZN BAM BBD BDT BGN BMD BND BOB BOV BRL BSD
BTN BWP BYN BZD CAD CDF CHE CHF CHW CNY COP COU CRC CUP CVE CZK DKK DOP DZD
EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD GTQ GYD HKD HNL HTG HUF IDR ILS
INR IRR JMD KES KGS KHR KPW KYD KZT LAK LBP LKR LRD LSL MAD MDL MGA MKD MMK
MNT MOP MRU MUR MVR MWK MXN MXV MYR MZN NAD NGN NIO NOK NPR NZD PAB PEN PGK
PHP PKR PLN QAR RON RSD RUB SAR SBD SCR SDG SEK SGD SHP SLE SOS SRD SSP STN
SVC SYP SZL THB TJS TMT TOP TRY TTD TWD TZS UAH USD USN UYU UZS VED VES WST
XAD XCD XCG YER ZAR ZMW ZWG
"""
CURRENCY_MINOR_UNITS: dict[str, int] = dict.fromkeys(_TWO_DECIMAL.split(), 2)
CURRENCY_MINOR_UNITS.update(
    dict.fromkeys(
        "BIF CLP DJF GNF ISK JPY KMF KRW PYG RWF UGX UYI VND VUV XAF XOF XPF".split(), 0
    )
)
CURRENCY_MINOR_UNITS.update(dict.fromkeys("BHD IQD JOD KWD LYD OMR TND".split(), 3))
CURRENCY_MINOR_UNITS.update(dict.fromkeys("CLF UYW".split(), 4))


def normalize_currency(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    code = value.strip().upper()
    return code if code in CURRENCY_MINOR_UNITS else None
