import os
import numpy as np
from scapy.all import rdpcap, Ether, IP, TCP, UDP, Raw
from collections import defaultdict

MTU         = 1500   # maximum transmission unit (bytes)
MAX_PACKETS = 10     # paper: truncate flows to 10 packets (20 for Tor)
UDP_PAD     = 12     # paper: pad UDP headers to match TCP (20 bytes)


# -----------------------------------
# Extract 5-tuple flow key
# -----------------------------------
def get_flow_key(pkt):
    if pkt.haslayer(IP) and (pkt.haslayer(TCP) or pkt.haslayer(UDP)):
        ip    = pkt[IP]
        proto = "TCP" if pkt.haslayer(TCP) else "UDP"
        return (ip.src, ip.dst, pkt.sport, pkt.dport, proto)
    return None


# -----------------------------------
# Strip Ethernet, mask IPs/ports,
# pad UDP, return raw bytes
# Paper section 4.2.1
# -----------------------------------
def preprocess_packet_bytes(pkt):
    pkt = pkt.copy()
    if pkt.haslayer(Ether):
        pkt = pkt[IP]
    if not pkt.haslayer(IP):
        return None

    ip     = pkt[IP]
    ip.src = "0.0.0.0"
    ip.dst = "0.0.0.0"

    if pkt.haslayer(TCP):
        tcp        = pkt[TCP]
        tcp.sport  = 0
        tcp.dport  = 0
        raw        = bytes(pkt)
    elif pkt.haslayer(UDP):
        udp        = pkt[UDP]
        udp.sport  = 0
        udp.dport  = 0
        raw        = bytes(pkt)
        ip_hlen    = ip.ihl * 4
        raw        = raw[:ip_hlen + 8] + b'\x00' * UDP_PAD + raw[ip_hlen + 8:]
    else:
        raw = bytes(pkt)

    return raw


# -----------------------------------
# Convert preprocessed bytes to
# a normalized 1500-dim float32 vector
# -----------------------------------
def packet_to_vector(pkt):
    raw = preprocess_packet_bytes(pkt)
    if raw is None:
        return None
    vec = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 255.0
    if len(vec) < MTU:
        vec = np.pad(vec, (0, MTU - len(vec)))
    else:
        vec = vec[:MTU]
    return vec


# -----------------------------------
# Raw packet size in bytes
# (original IP payload length)
# -----------------------------------
def get_packet_size(pkt) -> int:
    if pkt.haslayer(IP):
        return int(pkt[IP].len)
    return len(bytes(pkt))


# -----------------------------------
# Handshake / control packet filter
# -----------------------------------
def is_handshake_packet(pkt) -> bool:
    if pkt.haslayer(TCP):
        flags = pkt[TCP].flags
        if flags & 0x02 and not (flags & 0x10):  # SYN without ACK
            return True
        if flags & 0x04:                          # RST
            return True
    return False


# -----------------------------------
# Extract flows from one pcap file.
# Each flow entry is (pkt, timestamp, size).
# Returns dict: flow_key -> list of (pkt, float, int)
# -----------------------------------
def extract_flows_from_pcap(file_path, time_window=15, max_packets=MAX_PACKETS):
    print(f"  Reading: {os.path.basename(file_path)}")
    try:
        packets = rdpcap(file_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        return {}

    flows           = defaultdict(list)
    flow_start_time = {}

    for pkt in packets:
        key = get_flow_key(pkt)
        if key is None:
            continue
        if is_handshake_packet(pkt):
            continue

        t = float(pkt.time)
        s = get_packet_size(pkt)

        if key not in flow_start_time:
            flow_start_time[key] = t

        if t - flow_start_time[key] <= time_window:
            flows[key].append((pkt, t, s))
        else:
            flow_start_time[key] = t
            flows[key] = [(pkt, t, s)]

    # Truncate and drop short flows
    trimmed = {}
    for key, entries in flows.items():
        if len(entries) >= 2:
            trimmed[key] = entries[:max_packets]
    return trimmed


# -----------------------------------
# Convert flows to three aligned lists:
#
#   flow_vectors    list of np.array [N, 1500]   packet byte vectors
#   flow_timestamps list of np.array [N]          absolute timestamps
#   flow_sizes      list of np.array [N]          packet sizes (bytes)
#
# Index i in all three lists = same flow.
# -----------------------------------
def flows_to_vectors(flows):
    flow_vectors    = []
    flow_timestamps = []
    flow_sizes      = []

    for key, entries in flows.items():
        vecs, times, sizes = [], [], []

        for pkt, t, s in entries:
            try:
                vec = packet_to_vector(pkt)
                if vec is not None:
                    vecs.append(vec)
                    times.append(t)
                    sizes.append(float(s))
            except Exception:
                continue

        if len(vecs) >= 2:
            flow_vectors.append(np.array(vecs,  dtype=np.float32))
            flow_timestamps.append(np.array(times, dtype=np.float64))
            flow_sizes.append(np.array(sizes,  dtype=np.float32))

    return flow_vectors, flow_timestamps, flow_sizes


# -----------------------------------
# Smoke test
# -----------------------------------
if __name__ == "__main__":
    base    = os.path.join("data", "nonvpn")
    classes = [f for f in os.listdir(base) if os.path.isdir(os.path.join(base, f))]
    print("Classes:", classes)

    cls_path = os.path.join(base, classes[0])
    pcaps    = [f for f in os.listdir(cls_path) if f.endswith((".pcap", ".pcapng"))]
    flows    = extract_flows_from_pcap(os.path.join(cls_path, pcaps[0]))
    vecs, times, sizes = flows_to_vectors(flows)

    print(f"Valid flows : {len(vecs)}")
    if vecs:
        print(f"  vecs  : {vecs[0].shape}")
        print(f"  times : {times[0].shape}  range [{times[0].min():.3f}, {times[0].max():.3f}]")
        print(f"  sizes : {sizes[0].shape}  mean {sizes[0].mean():.1f} bytes")