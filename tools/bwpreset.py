"""Bitwig's .bwpreset container: read it into a tree, write a tree back.

    'BtWg' + '0003' + '0002' (plain; '0004' is encrypted) + 4 hex + data offset (8 hex) + 16 '0'
    then the metadata block (a 'meta' object of key/typed-value pairs), padded to the data offset
    with spaces and a newline, then the device: one object.

An object is a u32 class number, then fields (u32 field number, u8 type, the value) ended by a
u32 0. Objects count 1, 2, 3... in the order they start (the root is 1): a 'ref' field or an
'lref' list entry names an earlier object by that number.
"""
import struct

class Fail(Exception): pass

class R:
    def __init__(s, b, p=0): s.b, s.p = b, p
    def u8(s): v = s.b[s.p]; s.p += 1; return v
    def u16(s): v = struct.unpack('>H', s.b[s.p:s.p + 2])[0]; s.p += 2; return v
    def u32(s): v = struct.unpack('>I', s.b[s.p:s.p + 4])[0]; s.p += 4; return v
    def take(s, n): v = s.b[s.p:s.p + n]; s.p += n; return v


def read_string(r):
    n = r.u32()
    if n & 0x80000000:
        return ('u16', r.take(2 * (n & 0x7fffffff)).decode('utf-16-be'))
    return ('s', r.take(n).decode('utf-8'))


def read_value(r, t):
    if t == 0x01: return ('i8', r.u8())
    if t == 0x02: return ('i16', r.u16())
    if t == 0x03: return ('i32', r.u32())
    if t == 0x04: return ('i64', r.take(8))
    if t == 0x05: return ('bool', r.u8())
    if t == 0x06: return ('f32', r.take(4))
    if t == 0x07: return ('f64', struct.unpack('>d', r.take(8))[0])
    if t == 0x08: return read_string(r)
    if t == 0x09: return read_obj(r)
    if t == 0x0a: return ('null',)
    if t == 0x0b: return ('ref', r.u32())
    if t == 0x0d: n = r.u32(); return ('bytes', r.take(n))
    if t == 0x12:
        items = []
        while True:
            c = struct.unpack('>I', r.b[r.p:r.p + 4])[0]
            if c == 3: r.p += 4; break
            if c == 1: r.p += 4; items.append(('lref', r.u32())); continue
            items.append(read_obj(r))
        return ('list', items)
    if t == 0x14: n = r.u8(); return ('strmap', [(read_string(r), r.u32()) for _ in range(n)])
    if t == 0x15: return ('uuid', r.take(16))
    if t == 0x16: return ('b16', r.take(16))
    if t == 0x17: n = r.u32(); return ('f32s', r.take(4 * n))
    if t == 0x19: n = r.u32(); return ('strlist', [read_string(r) for _ in range(n)])
    raise Fail('type %02x at %x' % (t, r.p - 1))


def read_obj(r):
    cls = r.u32()
    fields = []
    while True:
        fid = r.u32()
        if fid == 0: break
        t = r.u8()
        fields.append([fid, t, read_value(r, t)])
    return ['obj', cls, fields]


def read(path_or_bytes):
    b = path_or_bytes if isinstance(path_or_bytes, bytes) else open(path_or_bytes, 'rb').read()
    if b[:4] != b'BtWg': raise Fail('not a Bitwig file')
    if b[8:12] != b'0002': raise Fail('encrypted (flag %s)' % b[8:12].decode())
    r = R(b, 0x2a)
    kind = r.u32(); name = read_string(r)
    meta = []
    while r.u32() == 1:
        k = read_string(r); t = r.u8(); meta.append([k, t, read_value(r, t)])
    off = int(b[16:24], 16)
    r = R(b, off)
    root = read_obj(r)
    if r.p != len(b): raise Fail('ended at %x of %x' % (r.p, len(b)))
    return {'head': b[:0x2a], 'kind': kind, 'name': name, 'meta': meta, 'root': root,
            'pad': b[0x2a:off]}


# ---- writing

def w_string(out, v):
    tag, s = v
    if tag == 'u16' or any(ord(c) > 127 for c in s):
        e = s.encode('utf-16-be'); out += struct.pack('>I', 0x80000000 | (len(e) // 2)) + e
    else:
        e = s.encode('utf-8'); out += struct.pack('>I', len(e)) + e


def number(root):
    """the 1-based start order of every object, by id()"""
    ids, n = {}, [0]
    def walk(o):
        n[0] += 1; ids[id(o)] = n[0]
        for f in o[2]: walk_v(f[2])
    def walk_v(v):
        if v[0] == 'obj': walk(v)
        elif v[0] == 'list':
            for it in v[1]:
                if it[0] == 'obj': walk(it)
    walk(root)
    return ids


def w_value(out, t, v, ids):
    k = v[0]
    if t == 0x01: out.append(v[1] & 0xff)
    elif t == 0x02: out += struct.pack('>H', v[1])
    elif t == 0x03: out += struct.pack('>I', v[1] & 0xffffffff)
    elif t in (0x04, 0x06, 0x15, 0x16): out += v[1]
    elif t == 0x05: out.append(1 if v[1] else 0)
    elif t == 0x07: out += struct.pack('>d', v[1])
    elif t == 0x08: w_string(out, v)
    elif t == 0x09: w_obj(out, v, ids)
    elif t == 0x0a: pass
    elif t == 0x0b: out += struct.pack('>I', v[1] if isinstance(v[1], int) else ids[id(v[1])])
    elif t == 0x0d: out += struct.pack('>I', len(v[1])) + v[1]
    elif t == 0x12:
        for it in v[1]:
            if it[0] == 'lref':
                out += struct.pack('>II', 1, it[1] if isinstance(it[1], int) else ids[id(it[1])])
            else: w_obj(out, it, ids)
        out += struct.pack('>I', 3)
    elif t == 0x14:
        out.append(len(v[1]))
        for s, n in v[1]: w_string(out, s); out += struct.pack('>I', n)
    elif t == 0x17: out += struct.pack('>I', len(v[1]) // 4) + v[1]
    elif t == 0x19:
        out += struct.pack('>I', len(v[1]))
        for s in v[1]: w_string(out, s)
    else: raise Fail('cannot write type %02x' % t)


def w_obj(out, o, ids):
    out += struct.pack('>I', o[1])
    for fid, t, v in o[2]:
        out += struct.pack('>IB', fid, t); w_value(out, t, v, ids)
    out += struct.pack('>I', 0)


def write(doc, path=None, pad=None):
    """doc as read() gives it; the metadata is padded as it was unless pad (bytes) is given"""
    meta = bytearray()
    meta += struct.pack('>I', doc['kind']); w_string(meta, doc['name'])
    for k, t, v in doc['meta']:
        meta += struct.pack('>I', 1); w_string(meta, k); meta.append(t); w_value(meta, t, v, {})
    meta += struct.pack('>I', 0)
    if pad is None:
        old = doc.get('pad', b'')
        pad = b'\n' if len(old) <= len(meta) else b' ' * (len(old) - len(meta) - 1) + b'\n'
    off = 0x2a + len(meta) + len(pad)
    head = doc['head'][:16] + b'%08x' % off + doc['head'][24:0x2a]
    data = bytearray()
    w_obj(data, doc['root'], number(doc['root']))
    b = bytes(head) + bytes(meta) + pad + bytes(data)
    if path: open(path, 'wb').write(b)
    return b


# ---- finding things in a tree

def field(o, fid, default=None):
    for f in o[2]:
        if f[0] == fid: return f[2]
    return default


def set_field(o, fid, t, v):
    for f in o[2]:
        if f[0] == fid: f[1], f[2] = t, v; return
    o[2].append([fid, t, v])


def named(o, name):
    """the object below o whose field 697 (its name) is name, depth first"""
    for fid, t, v in o[2]:
        for c in ([v] if v[0] == 'obj' else v[1] if v[0] == 'list' else []):
            if c[0] != 'obj': continue
            if field(c, 697, ('s', None))[1] == name: return c
            hit = named(c, name)
            if hit: return hit
    return None


def meta_get(doc, key):
    for k, t, v in doc['meta']:
        if k[1] == key: return v
    return None


def meta_set(doc, key, t, v):
    for m in doc['meta']:
        if m[0][1] == key: m[1], m[2] = t, v; return
    doc['meta'].append([('s', key), t, v]); doc['meta'].sort(key=lambda m: m[0][1])


# ---- building presets: Sampler (one sample, or zones) and Drum Machine (a Sampler a pad)
#
# The empty devices come from Bitwig's own default presets (its Library's device-settings, which
# are plain), so what we add is only the samples: a file (name, size, MD5 and where it is), a
# sample (rate, length, channels), and a zone or pad around it. The field numbers were read off
# presets and projects Bitwig wrote (5.3).

import copy, getpass, hashlib, os, socket, uuid

DRUM_MACHINE = '8ea97e45-0255-40fd-bc7e-94419741e9d1'
SAMPLER = '468bc14b-b2e7-45a1-9666-e83117fe404e'
LIBRARIES = [os.environ.get('BITWIG_LIBRARY', ''),
             '/opt/bitwig-studio/Library',
             '/usr/share/bitwig-studio/Library',
             '/Applications/Bitwig Studio.app/Contents/Resources/Library',
             r'C:\Program Files\Bitwig Studio\Library']


def library():
    for lib in LIBRARIES:
        if lib and os.path.isfile(os.path.join(lib, 'device-settings', SAMPLER, 'Default.bwpreset')):
            return lib
    return None


def default_preset(device):
    lib = library()
    if not lib: raise Fail('Bitwig Studio is not installed here (set BITWIG_LIBRARY to its Library folder)')
    return read(os.path.join(lib, 'device-settings', device, 'Default.bwpreset'))


def O(cls, *fields):
    """an object; fields are (number, type, value)"""
    return ['obj', cls, [list(f) for f in fields]]


def S(s): return ('u16', s) if any(ord(c) > 127 for c in s) else ('s', s)


def INT(fid, v):
    """an integer field the way Bitwig writes it: the narrowest type it fits"""
    if -128 <= v < 128: return (fid, 0x01, ('i8', v & 0xff))
    if -32768 <= v < 32768: return (fid, 0x02, ('i16', v & 0xffff))
    return (fid, 0x03, ('i32', v))


def machine_uuid():
    """what Bitwig files a local file's machine under: Java's nameUUIDFromBytes(/etc/machine-id)"""
    try: raw = open('/etc/machine-id', 'rb').read()
    except OSError: return ''
    h = bytearray(hashlib.md5(raw).digest()); h[6] = (h[6] & 0x0f) | 0x30; h[8] = (h[8] & 0x3f) | 0x80
    return str(uuid.UUID(bytes=bytes(h)))


def wav_facts(path):
    """(rate, frames, channels) from a WAV's fmt and data chunks"""
    with open(path, 'rb') as f: b = f.read()
    pos, rate, ch, align, frames = 12, 44100, 2, 4, 0
    while pos + 8 <= len(b):
        cid, size = b[pos:pos + 4], struct.unpack('<I', b[pos + 4:pos + 8])[0]
        if cid == b'fmt ':
            ch, rate = struct.unpack('<HI', b[pos + 10:pos + 16]); align = struct.unpack('<H', b[pos + 20:pos + 22])[0]
        elif cid == b'data': frames = size // max(align, 1)
        pos += 8 + size + (size & 1)
    return rate, frames, ch


def file_obj(path):
    """a sample file on disk (cls 1190): its name, where it is, MD5 and size"""
    path = os.path.abspath(path)
    data = open(path, 'rb').read()
    where = O(42, (56, 0x08, S(socket.gethostname())), (1229, 0x08, S(machine_uuid())),
              (58, 0x08, S(getpass.getuser())), (59, 0x08, S(path)))
    return O(1190, (1298, 0x08, S(os.path.basename(path))), (3386, 0x08, ('s', '')), (3284, 0x08, ('s', '')),
             (3387, 0x09, where), (21, 0x09, O(692, (3163, 0x12, ('list', [])))),
             (1299, 0x09, O(482, (22, 0x0d, ('bytes', hashlib.md5(data).digest())))),
             INT(23, len(data)), (4764, 0x12, ('list', [])))


def sample_obj(path, root=60, track=1.0, loop=None, files=None):
    """a sample as a Sampler plays it (cls 528): loop (start, end) in frames, or None for one-shot.
    files collects each file object, for the preset's own list of the files it uses."""
    rate, frames, ch = wav_facts(path)
    secs = frames / float(rate)
    f = file_obj(path)
    smp = O(1193, (4767, 0x08, S(os.path.basename(path))), (4766, 0x09, f), (4024, 0x08, ('s', '')),
            INT(148, rate), (149, 0x07, ('f64', secs)), INT(150, ch), (2237, 0x05, ('bool', 0)),
            (2238, 0x07, ('f64', 0.0)), (2239, 0x07, ('f64', 0.0)), INT(2240, -1), (2241, 0x07, ('f64', 0.0)),
            (10655, 0x05, ('bool', 0)))
    field(f, 4764)[1].append(('lref', smp))      # the file lists the samples that use it
    if files is not None: files.append(f)
    ls, le = (loop[0] / float(rate), loop[1] / float(rate)) if loop else (0.0, secs)
    return O(528, (697, 0x08, ('s', 'SAMPLE')), (1864, 0x09, smp), (1883, 0x07, ('f64', float(track))),
             INT(1884, root), (1885, 0x07, ('f64', 0.0)), (2534, 0x07, ('f64', 0.0)), (1886, 0x07, ('f64', 0.0)),
             (1887, 0x07, ('f64', secs)), (1888, 0x07, ('f64', ls)), (1889, 0x07, ('f64', le)),
             (1890, 0x07, ('f64', 0.0)), INT(1891, 1 if loop else 0), (1892, 0x05, ('bool', 0)),
             (10093, 0x07, ('f64', 0.0)), (10094, 0x07, ('f64', -1.0)))


def zone_obj(sample, lo, hi):
    return O(538, (697, 0x08, ('s', 'ZONE')), (1902, 0x09, sample), INT(9875, -1), INT(1903, lo), INT(1904, hi),
             INT(1905, 1), INT(1906, 127), INT(9878, 0), INT(9879, 127), INT(4006, 0), INT(4007, 0), INT(4008, 0),
             INT(4009, 0), INT(9880, 0), INT(9881, 0), INT(9882, 0), (10059, 0x07, ('f64', 0.0)),
             (10060, 0x07, ('f64', 0.0)), (10061, 0x07, ('f64', 0.0)))


def sampler_device(zones, files):
    """a Sampler device (the default's, cls 64) holding zones: (wav, low, high, root, track, loop).
    One zone over every key is a plain sample; more are a multisample."""
    doc = default_preset(SAMPLER)
    dev = copy.deepcopy(field(doc['root'], 5153))
    slot = named(dev, 'SAMPLE')
    if len(zones) == 1 and zones[0][1:3] == (0, 127):
        wav, lo, hi, root, track, loop = zones[0]
        set_field(slot, 1868, 0x09, sample_obj(wav, root, track, loop, files))
    else:
        zs = [zone_obj(sample_obj(wav, root, track, loop, files), lo, hi) for wav, lo, hi, root, track, loop in zones]
        set_field(slot, 1868, 0x09, O(537, (697, 0x08, ('s', 'MUTLISAMPLE')), (1901, 0x12, ('list', zs)),
                                      (4012, 0x12, ('list', [])), (4020, 0x08, ('s', '')), (4021, 0x08, ('s', '')),
                                      (4022, 0x08, ('s', '')), (4023, 0x08, ('s', ''))))
    return doc, dev


def finish(doc, files, category, creator, comment, tags, extra_meta=()):
    set_field(doc['root'], 5154, 0x12, ('list', [('lref', f) for f in files]))
    meta_set(doc, 'creator', 0x08, S(creator))
    meta_set(doc, 'comment', 0x08, S(comment))
    meta_set(doc, 'preset_category', 0x08, S(category))
    meta_set(doc, 'tags', 0x08, S(tags))
    for k, t, v in extra_meta: meta_set(doc, k, t, v)
    for k in ('revision_id', 'revision_no'):
        doc['meta'] = [m for m in doc['meta'] if m[0][1] != k]
    return doc


def sampler_preset(path, zones, category='', creator='soundmon', comment='', tags=''):
    files = []
    doc, dev = sampler_device(zones, files)
    set_field(doc['root'], 5153, 0x09, dev)
    write(finish(doc, files, category, creator, comment, tags), path, pad=b'\n')


def drum_preset(path, pads, creator='soundmon', comment='', tags=''):
    """pads: (key, name, wav) - each a Sampler with the one-shot on its own pad, no key tracking"""
    doc = default_preset(DRUM_MACHINE)
    files, objs, mods = [], [], []
    for key, name, wav in pads:
        sdoc, dev = sampler_device([(wav, 0, 127, 60, 0.0, None)], files)
        set_field(dev, 697, 0x08, ('s', '0'))
        mods = meta_get(sdoc, 'referenced_modulator_ids')
        mixer = O(570, (697, 0x08, ('s', 'MIXER')),
                  (2081, 0x09, O(133, (697, 0x08, ('s', 'VOLUME')), (310, 0x07, ('f64', 1.0)))),
                  (2082, 0x09, O(133, (697, 0x08, ('s', 'PANORAMA')), (310, 0x07, ('f64', 0.0)))),
                  (2083, 0x09, O(127, (697, 0x08, ('s', 'MUTE')), (303, 0x05, ('bool', 0)))),
                  (3026, 0x12, ('list', [])))
        objs.append(O(594, (697, 0x08, ('s', 'PAD%d' % key)), (10936, 0x15, ('uuid', bytes(16))),
                      (2271, 0x08, S(name)),
                      (841, 0x09, O(51, (697, 0x08, ('s', '')), (135, 0x12, ('list', [dev])))),
                      (2085, 0x09, mixer), (4993, 0x05, ('bool', 1)), (5381, 0x16, ('b16', bytes(16))),
                      INT(9946, -1), INT(9947, -1), INT(2277, key)))
    set_field(named(doc['root'], 'DRUM_PADS'), 2272, 0x12, ('list', objs))
    extra = [('referenced_device_ids', 0x19, ('strlist', [('s', DRUM_MACHINE), ('s', SAMPLER)]))]
    if mods: extra.append(('referenced_modulator_ids', 0x19, mods))
    write(finish(doc, files, 'Drums', creator, comment, tags, extra), path, pad=b'\n')


if __name__ == '__main__':
    import sys
    ok = bad = 0
    for f in sys.argv[1:]:
        try:
            b = open(f, 'rb').read()
            if b[8:12] != b'0002': continue
            if write(read(b)) != b: raise Fail('differs')
            ok += 1
        except Exception as e:
            bad += 1; print('FAIL', f, e)
    print('round trip ok', ok, 'bad', bad)
