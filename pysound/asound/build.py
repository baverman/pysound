from cffi import FFI
ffibuilder = FFI()

ffibuilder.set_source("pysound.asound._asound", """\
    #include "alsa/asoundlib.h"
""", libraries=['asound'])

ffibuilder.cdef("""\
typedef ... snd_seq_t;
typedef ... snd_seq_port_subscribe_t;

typedef struct snd_seq_addr {
	unsigned char client;	/**< Client id */
	unsigned char port;	/**< Port id */
} snd_seq_addr_t;

typedef unsigned char snd_seq_event_type_t;

enum snd_seq_event_type {
	/** note on and off with duration; event data type = #snd_seq_ev_note_t */
	SND_SEQ_EVENT_NOTE = 5,
	/** note on; event data type = #snd_seq_ev_note_t */
	SND_SEQ_EVENT_NOTEON,
	/** note off; event data type = #snd_seq_ev_note_t */
	SND_SEQ_EVENT_NOTEOFF,
	/** key pressure change (aftertouch); event data type = #snd_seq_ev_note_t */
	SND_SEQ_EVENT_KEYPRESS,

	/** controller; event data type = #snd_seq_ev_ctrl_t */
	SND_SEQ_EVENT_CONTROLLER = 10,
	/** program change; event data type = #snd_seq_ev_ctrl_t */
	SND_SEQ_EVENT_PGMCHANGE,
	/** channel pressure; event data type = #snd_seq_ev_ctrl_t */
	SND_SEQ_EVENT_CHANPRESS,
	/** pitchwheel; event data type = #snd_seq_ev_ctrl_t; data is from -8192 to 8191) */
	SND_SEQ_EVENT_PITCHBEND,
	/** 14 bit controller value; event data type = #snd_seq_ev_ctrl_t */
	SND_SEQ_EVENT_CONTROL14,
	/** 14 bit NRPN;  event data type = #snd_seq_ev_ctrl_t */
	SND_SEQ_EVENT_NONREGPARAM,
	/** 14 bit RPN; event data type = #snd_seq_ev_ctrl_t */
	SND_SEQ_EVENT_REGPARAM
};

/** Real-time data record */
typedef struct snd_seq_real_time {
	unsigned int tv_sec;		/**< seconds */
	unsigned int tv_nsec;		/**< nanoseconds */
} snd_seq_real_time_t;

/** (MIDI) Tick-time data record */
typedef unsigned int snd_seq_tick_time_t;

/** unioned time stamp */
typedef union snd_seq_timestamp {
	snd_seq_tick_time_t tick;	/**< tick-time */
	struct snd_seq_real_time time;	/**< real-time */
} snd_seq_timestamp_t;

/** Note event */
typedef struct snd_seq_ev_note {
	unsigned char channel;		/**< channel number */
	unsigned char note;		/**< note */
	unsigned char velocity;		/**< velocity */
	unsigned char off_velocity;	/**< note-off velocity; only for #SND_SEQ_EVENT_NOTE */
	unsigned int duration;		/**< duration until note-off; only for #SND_SEQ_EVENT_NOTE */
} snd_seq_ev_note_t;

/** Controller event */
typedef struct snd_seq_ev_ctrl {
	unsigned char channel;		/**< channel number */
	unsigned char unused[3];	/**< reserved */
	unsigned int param;		/**< control parameter */
	signed int value;		/**< control value */
} snd_seq_ev_ctrl_t;

typedef struct snd_seq_event {
	snd_seq_event_type_t type;	/**< event type */
	unsigned char flags;		/**< event flags */
	unsigned char tag;		/**< tag */
	
	unsigned char queue;		/**< schedule queue */
	snd_seq_timestamp_t time;	/**< schedule time */

	snd_seq_addr_t source;		/**< source address */
	snd_seq_addr_t dest;		/**< destination address */

	union {
		snd_seq_ev_note_t note;		/**< note information */
		snd_seq_ev_ctrl_t control;	/**< MIDI control information */
	} data;				/**< event data... */
} snd_seq_event_t;

#define SND_SEQ_OPEN_OUTPUT ...
#define SND_SEQ_OPEN_INPUT ...
#define SND_SEQ_OPEN_DUPLEX ...
#define SND_SEQ_PORT_SYSTEM_TIMER ...
#define SND_SEQ_PORT_SYSTEM_ANNOUNCE ...

#define SND_SEQ_PORT_CAP_READ ...
#define SND_SEQ_PORT_CAP_WRITE ...

#define SND_SEQ_PORT_CAP_SYNC_READ ...
#define SND_SEQ_PORT_CAP_SYNC_WRITE ...

#define SND_SEQ_PORT_CAP_DUPLEX	...

#define SND_SEQ_PORT_CAP_SUBS_READ ...
#define SND_SEQ_PORT_CAP_SUBS_WRITE ...
#define SND_SEQ_PORT_CAP_NO_EXPORT ...

#define SND_SEQ_PORT_TYPE_SPECIFIC ...
#define SND_SEQ_PORT_TYPE_MIDI_GENERIC ...
#define SND_SEQ_PORT_TYPE_MIDI_GM ...
#define SND_SEQ_PORT_TYPE_MIDI_GS ...
#define SND_SEQ_PORT_TYPE_MIDI_XG ...
#define SND_SEQ_PORT_TYPE_MIDI_MT32 ...
#define SND_SEQ_PORT_TYPE_MIDI_GM2 ...
#define SND_SEQ_PORT_TYPE_SYNTH	...
#define SND_SEQ_PORT_TYPE_DIRECT_SAMPLE ...
#define SND_SEQ_PORT_TYPE_SAMPLE ...
#define SND_SEQ_PORT_TYPE_HARDWARE ...
#define SND_SEQ_PORT_TYPE_SOFTWARE ...
#define SND_SEQ_PORT_TYPE_SYNTHESIZER ...
#define SND_SEQ_PORT_TYPE_PORT ...
#define SND_SEQ_PORT_TYPE_APPLICATION ...

int snd_seq_open(snd_seq_t **handle, const char *name, int streams, int mode);
int snd_seq_set_client_name(snd_seq_t *seq, const char *name);
int snd_seq_create_simple_port(snd_seq_t *seq, const char *name,
			       unsigned int caps, unsigned int type);
int snd_seq_event_input(snd_seq_t *handle, snd_seq_event_t **ev);
const char* snd_seq_name (snd_seq_t *seq);

int snd_seq_client_id(snd_seq_t *handle);

int snd_seq_port_subscribe_malloc(snd_seq_port_subscribe_t **ptr);
void snd_seq_port_subscribe_set_sender(snd_seq_port_subscribe_t *info, const snd_seq_addr_t *addr);
void snd_seq_port_subscribe_set_dest(snd_seq_port_subscribe_t *info, const snd_seq_addr_t *addr);
void snd_seq_port_subscribe_set_queue(snd_seq_port_subscribe_t *info, int q);
void snd_seq_port_subscribe_set_exclusive(snd_seq_port_subscribe_t *info, int val);
void snd_seq_port_subscribe_set_time_update(snd_seq_port_subscribe_t *info, int val);
void snd_seq_port_subscribe_set_time_real(snd_seq_port_subscribe_t *info, int val);

int snd_seq_get_port_subscription(snd_seq_t *handle, snd_seq_port_subscribe_t *sub);
int snd_seq_subscribe_port(snd_seq_t *handle, snd_seq_port_subscribe_t *sub);
int snd_seq_unsubscribe_port(snd_seq_t *handle, snd_seq_port_subscribe_t *sub);
""")

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
