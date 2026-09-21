#!/usr/bin/env python3
"""Reproducible DOCX extraction including table order and math accents/subscripts."""
import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as E
from zipfile import ZipFile
N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}

def content(e):
    tag=e.tag.split('}')[-1]
    if tag=='t': return e.text or ''
    if tag=='acc':
        char=e.find('m:accPr/m:chr',N).get('{'+N['m']+'}val')
        return content(e.find('m:e',N))+char
    if tag=='sSub': return content(e.find('m:e',N))+'_'+content(e.find('m:sub',N))
    if tag in ['accPr','rPr','pPr']: return ''
    return ''.join(content(x) for x in e)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('docx'); p.add_argument('--out',required=True); a=p.parse_args()
    path=Path(a.docx)
    with ZipFile(path) as z: r=E.fromstring(z.read('word/document.xml'))
    paragraphs=r.findall('.//w:body//w:p',N)
    Path(a.out).write_text('\n'.join(content(p) for p in paragraphs)+'\n')
    Path(a.out+'.json').write_text(json.dumps({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'paragraphs':len(paragraphs),'tables':len(r.findall('.//w:tbl',N)),
          'math_objects':len(r.findall('.//m:oMath',N))},indent=2)+'\n')
