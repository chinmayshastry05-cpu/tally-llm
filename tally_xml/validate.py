"""Structural validator for TallyPrime XML import messages.

Checks a structural SUBSET of the Tally schema — well-formed XML, the
expected ENVELOPE/voucher-or-ledger shape, balanced debits/credits, and
field consistency against a gold record when one is supplied. This is NOT a
complete Tally schema validator: it does not check the full Tally field
catalogue, data types, or company-specific masters. Returns 1/0 per check.

KNOWN HOLE (kept for backwards compatibility — GUI/samples use this):
``check()`` accepts impossible dates (e.g. 20261399 passes: the date is
only an 8-digit regex), uses float money, and does not verify ledger
presence or voucher-type allowlists. For the PROTOTYPE ACCEPTANCE gate,
use ``tally_xml.strict.validate_strict`` (real calendar dates, Decimal
money, caller-provided known ledgers, fail-closed). ``check()`` is
research only — never the acceptance verdict or download gate. Do not
use ``check()`` to approve a transaction for import.
"""
import re,xml.etree.ElementTree as ET
def check(x,gold=None):
    r={'wellformed':0,'schema':0,'balanced':0,'fields':0}
    x=x.strip()
    m=re.search(r'<ENVELOPE>.*</ENVELOPE>',x,re.S)
    if m: x=m.group(0)
    try: root=ET.fromstring(x)
    except Exception: return r
    r['wellformed']=1
    g=lambda p:(root.findtext(p) or '').strip()
    if root.tag!='ENVELOPE' or g('HEADER/TALLYREQUEST')!='Import' or g('HEADER/TYPE')!='Data': return r
    tm=root.find('BODY/DATA/TALLYMESSAGE')
    if tm is None: return r
    v=tm.find('VOUCHER'); l=tm.find('LEDGER')
    if l is not None:
        ok=g('HEADER/ID')=='All Masters' and l.get('NAME') and l.findtext('PARENT') and l.findtext('NAME.LIST/NAME')==l.get('NAME')
        r['schema']=r['balanced']=int(bool(ok))
        if gold: r['fields']=int(bool(ok) and gold['kind']=='ledger' and l.get('NAME')==gold['name'] and l.findtext('PARENT')==gold['parent'])
        return r
    if v is None: return r
    es=v.findall('ALLLEDGERENTRIES.LIST'); ds=(v.findtext('DATE') or '')
    ok=(g('HEADER/ID')=='Vouchers' and v.get('VCHTYPE')==v.findtext('VOUCHERTYPENAME') and v.get('ACTION')=='Create' and re.fullmatch(r'\d{8}',ds) and len(es)>=2 and all(e.findtext('LEDGERNAME') and e.findtext('ISDEEMEDPOSITIVE') in('Yes','No') for e in es))
    if not ok: return r
    r['schema']=1
    try:
        am=[float(e.findtext('AMOUNT')) for e in es]
        signs=all((a<0)==(e.findtext('ISDEEMEDPOSITIVE')=='Yes') for a,e in zip(am,es))
        r['balanced']=int(abs(sum(am))<1e-6 and signs)
        if gold and gold['kind']=='voucher' and r['balanced']:
            dr=[e.findtext('LEDGERNAME') for e,a in zip(es,am) if a<0]; cr=[e.findtext('LEDGERNAME') for e,a in zip(es,am) if a>0]
            r['fields']=int(v.get('VCHTYPE')==gold['vt'] and ds==gold['date'] and dr==[gold['dr']] and cr==[gold['cr']] and abs(max(abs(a) for a in am)-gold['amount'])<1e-6)
    except Exception: pass
    return r
