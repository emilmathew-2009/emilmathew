import os, uuid, threading, shutil, subprocess, tempfile, math
from pathlib import Path
import numpy as np
import librosa
from fastapi import UploadFile, File, HTTPException
from basic_pitch.inference import predict, ICASSP_2022_MODEL_PATH

AUDIO_ROOT=Path(os.getenv('AUDIO_WORKDIR','/tmp/emil-audio')); AUDIO_ROOT.mkdir(parents=True,exist_ok=True)
audio_jobs={}; audio_lock=threading.Lock()
NAMES=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
MAJOR=[0,4,7]; MINOR=[0,3,7];

def midi_name(m): return NAMES[int(m)%12]+str(int(m)//12-1)
def estimate_key(chroma):
    prof=np.mean(chroma,axis=1); best=(-1,None)
    for root in range(12):
        for mode,pat in [('major',MAJOR),('minor',MINOR)]:
            score=sum(prof[(root+x)%12] for x in pat)/(sum(prof)+1e-9)
            if score>best[0]: best=(score,(root,mode))
    return f'{NAMES[best[1][0]]} {"Major" if best[1][1]=="major" else "Minor"}'

def chord_for(vec, root):
    # Compare simple major/minor/dim/7th templates against the local chroma vector.
    candidates=[]
    templates=[('',[0,4,7]),('m',[0,3,7]),('dim',[0,3,6]),('7',[0,4,7,10]),('m7',[0,3,7,10])]
    v=vec/(np.linalg.norm(vec)+1e-9)
    for r in range(12):
        for suffix,ints in templates:
            t=np.zeros(12); t[[(r+i)%12 for i in ints]]=1
            t=t/(np.linalg.norm(t)+1e-9)
            candidates.append((float(np.dot(v,t)),r,suffix,ints))
    return max(candidates,key=lambda x:x[0])

def chord_name(root,suffix): return NAMES[root]+suffix

def infer_chords(y,sr,duration,tempo):
    hop=512; chroma=librosa.feature.chroma_cqt(y=y,sr=sr,hop_length=hop)
    frame_sec=hop/sr; step=max(0.5,60.0/(tempo or 90.0)*2)
    frames=max(1,int(step/frame_sec)); chords=[]
    for a in range(0,chroma.shape[1],frames):
        b=min(chroma.shape[1],a+frames); vec=np.mean(chroma[:,a:b],axis=1); score,root,suffix,ints=chord_for(vec,root=0)
        if score<0.35: continue
        start=a*frame_sec; end=min(duration,b*frame_sec)
        notes=[]
        # Keep left hand comfortably below middle C.
        base=48+root
        if base>55: base-=12
        for i in ints: notes.append(base+i)
        name=chord_name(root,suffix)
        if chords and chords[-1]['name']==name and abs(chords[-1]['start']-start-chords[-1]['duration'])<0.1:
            chords[-1]['duration']=end-chords[-1]['start']
        else: chords.append({'start':start,'duration':max(0.25,end-start),'name':name,'notes':notes})
    return chords

def extract_melody(midi_data):
    notes=[]
    for inst in getattr(midi_data,'instruments',[]):
        for n in inst.notes:
            notes.append((float(n.start),float(n.end),int(n.pitch),float(getattr(n,'velocity',80))))
    if not notes: return []
    notes.sort()
    # At each onset cluster, choose the highest sustained note; this favors lead vocals/melody.
    selected=[]
    for s,e,p,v in notes:
        overlapping=[x for x in notes if x[0] <= s+0.045 and x[1] >= s+0.04]
        if overlapping and p != max(x[2] for x in overlapping): continue
        if selected and p==selected[-1]['midi'] and s <= selected[-1]['start']+selected[-1]['duration']+0.08:
            selected[-1]['duration']=max(selected[-1]['duration'],e-selected[-1]['start'])
        else:
            selected.append({'start':s,'duration':max(0.06,e-s),'midi':p,'name':midi_name(p)})
    return selected

def worker(job_id,infile):
    try:
        y,sr=librosa.load(infile,sr=22050,mono=True)
        if len(y)<sr//2: raise RuntimeError('Audio is too short to analyze.')
        duration=len(y)/sr
        tempo_arr,_=librosa.beat.beat_track(y=y,sr=sr)
        tempo=float(np.asarray(tempo_arr).reshape(-1)[0]) if np.size(tempo_arr) else 90.0
        if not 35<=tempo<=220: tempo=90.0
        chroma=librosa.feature.chroma_cqt(y=y,sr=sr)
        key=estimate_key(chroma)
        _,midi_data,_=predict(str(infile),ICASSP_2022_MODEL_PATH)
        melody=extract_melody(midi_data)
        chords=infer_chords(y,sr,duration,tempo)
        score={'key':key,'tempo':round(tempo,1),'duration':round(duration,2),'melody':melody,'chords':chords}
        with audio_lock: audio_jobs[job_id]={'status':'done','score':score}
    except Exception as e:
        with audio_lock: audio_jobs[job_id]={'status':'error','error':str(e)}
    finally:
        try: infile.unlink(missing_ok=True)
        except: pass

def register_audio_routes(app):
    @app.post('/audio-jobs')
    async def create_audio_job(file: UploadFile=File(...)):
        name=(file.filename or 'audio').lower()
        if not name.endswith(('.mp3','.wav','.m4a','.ogg','.flac','.webm','.aac')): raise HTTPException(400,'Supported audio: MP3, WAV, M4A, OGG, FLAC, WEBM, AAC')
        data=await file.read()
        if len(data)>150*1024*1024: raise HTTPException(413,'Audio file too large (150 MB maximum).')
        job_id=uuid.uuid4().hex; d=AUDIO_ROOT/job_id; d.mkdir(parents=True,exist_ok=True); infile=d/('input'+Path(name).suffix); infile.write_bytes(data)
        with audio_lock: audio_jobs[job_id]={'status':'processing'}
        threading.Thread(target=worker,args=(job_id,infile),daemon=True).start()
        return {'jobId':job_id}
    @app.get('/audio-jobs/{job_id}')
    def audio_job(job_id:str):
        with audio_lock: j=audio_jobs.get(job_id)
        if not j: raise HTTPException(404,'Audio job not found')
        return j
