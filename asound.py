from _asound import ffi, lib
NULL = ffi.NULL


def capture_keyboard(handle, port, source):
    subs_p = ffi.new('snd_seq_port_subscribe_t **')
    lib.snd_seq_port_subscribe_malloc(subs_p)
    subs = subs_p[0]

    sclient, _, sport = source.partition(':')
    sender = ffi.new('snd_seq_addr_t *')
    sender.client = int(sclient)
    sender.port = int(sport)

    dest = ffi.new('snd_seq_addr_t *')
    dest.client = lib.snd_seq_client_id(handle)
    dest.port = port

    lib.snd_seq_port_subscribe_set_sender(subs, sender)
    lib.snd_seq_port_subscribe_set_dest(subs, dest)
    lib.snd_seq_port_subscribe_set_queue(subs, 1)
    lib.snd_seq_port_subscribe_set_time_update(subs, 1)
    lib.snd_seq_port_subscribe_set_time_real(subs, 1)
    lib.snd_seq_subscribe_port(handle, subs)
    return subs_p, sender, dest


def todict(data):
    result = {}
    for k in dir(data):
        if k[0] != '_':
            result[k] = getattr(data, k)
    return result


def listen(source, cb):
    p_handle = ffi.new('snd_seq_t**')
    lib.snd_seq_open(p_handle, b"default", lib.SND_SEQ_OPEN_INPUT, 0)
    handle = p_handle[0]

    lib.snd_seq_set_client_name(handle, b"Midi Listener");

    in_port = lib.snd_seq_create_simple_port(handle, b"listen:in",
                      lib.SND_SEQ_PORT_CAP_WRITE|lib.SND_SEQ_PORT_CAP_SUBS_WRITE,
                      lib.SND_SEQ_PORT_TYPE_APPLICATION)

    info = capture_keyboard(handle, in_port, source)

    ev = ffi.new('snd_seq_event_t**')
    onoff = (lib.SND_SEQ_EVENT_NOTEOFF, lib.SND_SEQ_EVENT_NOTEON)
    while True:
        lib.snd_seq_event_input(handle, ev);
        t = ev[0].type
        if t in onoff:
            e = ev[0].data.note
            cb(onoff.index(t), (e.channel, e.note))
        elif t == lib.SND_SEQ_EVENT_CONTROLLER:
            e = ev[0].data.control
            cb(3, (e.channel, e.param, e.value))
