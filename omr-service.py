import os, uuid, zipfile, tempfile, shutil, subprocess, threading
from pathlib import Path
from xml.etree import ElementTree as ET
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from audio_transcriber import register_audio_routes

app = FastAPI(title='Emil Sheet Music OMR')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
ROOT=Path(os.getenv('OMR_WORKDIR','/tmp/emil-omr')); ROOT.mkdir(parents=True, exist_ok=True)
jobs={}; lock=threading.Lock()

def local(tag): return tag.rsplit('}',1)[-1]

def parse_pitch(note):
    p=note.find('./pitch')
    if p is None: return None
    step=p.findtext('step'); octave=p.findtext('octave')
    alter=p.findtext('alter','0')
    if not step or octave is None: return None
    try: alter=int(float(alter)); octave=int(octave)
    except: return None
    midi={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[step]+12*(octave+1)+alter
    return midi

def read_musicxml(path):
    raw=path.read_bytes()
    if path.suffix.lower()=='.mxl':
        with zipfile.ZipFile(path) as z:
            names=z.namelist(); target=None
            for n in names:
                if n.endswith('container.xml'):
                    try:
                        root=ET.fromstring(z.read(n))
                        for el in root.iter():
                            if local(el.tag)=='rootfile':
                                target=el.attrib.get('full-path'); break
                    except: pass
                if target: break
            if not target:
                target=next((n for n in names if n.lower().endswith('.xml') and 'META-INF' not in n),None)
            if not target: raise ValueError('No MusicXML document found in MXL')
            raw=z.read(target)
    root=ET.fromstring(raw)
    # Normalize MusicXML namespaces so the parser works with both namespaced and non-namespaced files.
    for el in root.iter():
        el.tag=local(el.tag)
    parts=[]
    for score_part in root.findall('.//score-part'):
        pid=score_part.attrib.get('id','P1'); name=score_part.findtext('part-name') or pid
        parts.append((pid,name))
    part_names=dict(parts)
    events=[]; divisions=1; tempo=None; key=None; title=root.findtext('.//work/work-title') or root.findtext('.//movement-title') or 'Sheet Music'
    for part in root.findall('./part'):
        pid=part.attrib.get('id','P1'); cursor=0.0; last_start=0.0; part_div=divisions
        for measure in part.findall('./measure'):
            attrs=measure.find('attributes')
            if attrs is not None:
                d=attrs.findtext('divisions')
                if d:
                    try: part_div=float(d)
                    except: pass
                ks=attrs.find('key');
                if ks is not None and key is None:
                    fifths=ks.findtext('fifths'); mode=ks.findtext('mode') or 'major'
                    key=f'{fifths} fifths ({mode})'
            direction=measure.find('.//direction')
            if direction is not None and tempo is None:
                tm=direction.find('.//sound')
                if tm is not None and tm.get('tempo'):
                    try: tempo=float(tm.get('tempo'))
                    except: pass
                pm=direction.find('.//per-minute')
                if pm is not None:
                    try: tempo=float(pm.text)
                    except: pass
            for child in measure:
                tag=local(child.tag)
                if tag=='backup':
                    dur=float(child.findtext('duration','0') or 0); cursor-=dur/part_div
                    continue
                if tag=='forward':
                    dur=float(child.findtext('duration','0') or 0); cursor+=dur/part_div
                    continue
                if tag!='note': continue
                dur=float(child.findtext('duration','0') or 0)/part_div
                is_chord=child.find('chord') is not None
                start=last_start if is_chord else cursor
                midi=parse_pitch(child)
                if midi is not None:
                    staff=child.findtext('staff') or '1'; voice=child.findtext('voice') or '1'
                    events.append({'beat':start,'duration':dur,'midi':midi,'part':part_names.get(pid,pid),'staff':staff,'voice':voice})
                if not is_chord:
                    last_start=cursor; cursor+=dur
            # measure can have barline but cursor is authoritative from durations
    if events:
        minbeat=min(e['beat'] for e in events)
        if minbeat<0:
            for e in events: e['beat']-=minbeat
    return {'title':title,'key':key or 'Not detected','tempo':tempo,'events':events,'musicxml':raw.decode('utf-8')}

def worker(job_id, infile):
    jobdir=ROOT/job_id; out=jobdir/'out'; out.mkdir(parents=True,exist_ok=True)
    try:
        cmd=['/opt/audiveris/bin/Audiveris','-batch','-transcribe','-export','-output',str(out),str(infile)]
        env=os.environ.copy(); env.setdefault('JAVA_TOOL_OPTIONS','-Djava.awt.headless=true')
        p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=int(os.getenv('OMR_TIMEOUT','300')),env=env)
        candidates=list(out.rglob('*.mxl'))+list(out.rglob('*.xml'))
        candidates=[x for x in candidates if x.suffix.lower()=='.mxl' or 'container.xml' not in x.name]
        if p.returncode!=0 or not candidates:
            raise RuntimeError('Audiveris failed: '+p.stdout[-5000:])
        score=read_musicxml(candidates[0])
        with lock: jobs[job_id]={'status':'done','score':score,'log':p.stdout[-4000:]}
    except Exception as e:
        with lock: jobs[job_id]={'status':'error','error':str(e)}
    finally:
        try: infile.unlink(missing_ok=True)
        except: pass

@app.get('/health')
def health(): return {'ok':True,'engine':'Audiveris','audio_engine':'Basic Pitch + librosa','version':os.getenv('AUDIVERIS_VERSION','5.11.0')}

register_audio_routes(app)

@app.post('/jobs')
async def create_job(file: UploadFile=File(...)):
    name=(file.filename or 'score').lower()
    if not name.endswith(('.pdf','.png','.jpg','.jpeg','.webp')): raise HTTPException(400,'Supported formats: PDF, PNG, JPG, JPEG, WEBP')
    job_id=uuid.uuid4().hex; jobdir=ROOT/job_id; jobdir.mkdir(parents=True,exist_ok=True)
    suffix=Path(name).suffix or '.pdf'; infile=jobdir/('input'+suffix)
    data=await file.read()
    if len(data)>50*1024*1024: shutil.rmtree(jobdir,ignore_errors=True); raise HTTPException(413,'File too large (50 MB maximum).')
    infile.write_bytes(data)
    with lock: jobs[job_id]={'status':'processing'}
    threading.Thread(target=worker,args=(job_id,infile),daemon=True).start()
    return {'jobId':job_id}

@app.get('/jobs/{job_id}')
def job(job_id:str):
    with lock: j=jobs.get(job_id)
    if not j: raise HTTPException(404,'Job not found')
    return j
