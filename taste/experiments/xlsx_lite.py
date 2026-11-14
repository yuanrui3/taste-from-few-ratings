"""Read the first sheet of an .xlsx file with the standard library only (no openpyxl on the cluster)."""
import re
import zipfile
import xml.etree.ElementTree as ET

_NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def read_rows(path):
    """Rows of the first sheet as dicts {column letter: text}, header row included."""
    z = zipfile.ZipFile(path); ss = []
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', _NS):
            ss.append(''.join(t.text or '' for t in si.iter('{%s}t' % _NS['m'])))
    sheet = sorted(n for n in z.namelist() if n.startswith('xl/worksheets/sheet'))[0]
    rows = []
    for r in ET.fromstring(z.read(sheet)).iter('{%s}row' % _NS['m']):
        row = {}
        for c in r.findall('m:c', _NS):
            v = c.find('m:v', _NS)
            if v is not None:
                row[re.match(r'[A-Z]+', c.get('r')).group()] = ss[int(v.text)] if c.get('t') == 's' else v.text
        rows.append(row)
    return rows


def table(path):
    """Header names -> list of column values (None where empty)."""
    rows = read_rows(path); head = rows[0]
    return {name: [r.get(col) for r in rows[1:]] for col, name in head.items()}
