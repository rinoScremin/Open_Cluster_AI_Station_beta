#!/bin/bash
set -euo pipefail

# Allow explicit override (no extra stdout).
if [[ -n "${FORCE_LAN_IP:-}" ]]; then
    echo "${FORCE_LAN_IP}"
    exit 0
fi

# Define common wired interface prefixes, now including 'eno'
WIRED_PREFIXES="^(enx|eth|enp|eno)"

get_ipv4() {
    local iface="$1"
    # Prefer global IPv4 addresses; take the first one.
    ip -4 -o addr show dev "$iface" scope global | awk '{print $4}' | cut -d/ -f1 | head -n 1
}

iface_is_up() {
    local iface="$1"
    local oper="down"
    local carrier="0"
    if [[ -f "/sys/class/net/$iface/operstate" ]]; then
        oper="$(cat "/sys/class/net/$iface/operstate")"
    fi
    if [[ -f "/sys/class/net/$iface/carrier" ]]; then
        carrier="$(cat "/sys/class/net/$iface/carrier")"
    fi
    [[ "$oper" == "up" && "$carrier" == "1" ]]
}

# 1) If a specific interface is requested, use it.
if [[ -n "${FORCE_LAN_IFACE:-}" ]]; then
    if ip link show dev "${FORCE_LAN_IFACE}" >/dev/null 2>&1; then
        LAN_IP="$(get_ipv4 "${FORCE_LAN_IFACE}")"
        if [[ -n "$LAN_IP" ]]; then
            echo "$LAN_IP"
            exit 0
        fi
    fi
    echo "Error: FORCE_LAN_IFACE='${FORCE_LAN_IFACE}' has no IPv4 address." >&2
    exit 1
fi

# 2) Prefer the default route interface if it's wired.
DEFAULT_DEV="$(ip -4 route show default 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')"
if [[ -n "${DEFAULT_DEV:-}" && "$DEFAULT_DEV" =~ $WIRED_PREFIXES ]]; then
    LAN_IP="$(get_ipv4 "$DEFAULT_DEV")"
    if [[ -n "$LAN_IP" ]]; then
        echo "$LAN_IP"
        exit 0
    fi
fi

# 3) Scan wired interfaces that are up and have carrier; pick the one with
# the lowest default-route metric (if any), otherwise the first valid one.
best_iface=""
best_metric=""
for iface in /sys/class/net/*; do
    iface="$(basename "$iface")"
    [[ "$iface" =~ $WIRED_PREFIXES ]] || continue
    iface_is_up "$iface" || continue
    LAN_IP="$(get_ipv4 "$iface")"
    [[ -n "$LAN_IP" ]] || continue

    metric="$(ip -4 route show default dev "$iface" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="metric"){print $(i+1); exit}}')"
    metric="${metric:-9999}"
    if [[ -z "$best_iface" || "$metric" -lt "$best_metric" ]]; then
        best_iface="$iface"
        best_metric="$metric"
        best_ip="$LAN_IP"
    fi
done

if [[ -n "${best_iface:-}" ]]; then
    echo "$best_ip"
    exit 0
fi

# If the loop finishes without finding a valid IP
echo "Error: No active wired LAN interface found with an IPv4 address." >&2
exit 1
