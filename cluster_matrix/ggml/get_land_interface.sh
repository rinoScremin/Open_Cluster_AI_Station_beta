#!/bin/bash

# Define common wired interface prefixes, now including 'eno'
WIRED_PREFIXES="enx|eth|enp|eno"

# Use a loop to check active interfaces for a match
for iface in $(ls /sys/class/net/); do
    # Check if the interface name matches one of the wired prefixes
    if [[ "$iface" =~ $WIRED_PREFIXES ]]; then
        # Check if the interface is UP and RUNNING
        if [[ "$(cat /sys/class/net/$iface/operstate)" == "up" ]]; then
            # Extract the IPv4 address for this specific interface
            LAN_IP=$(ip -4 addr show dev "$iface" | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)
            
            if [ -n "$LAN_IP" ]; then
                # Found the wired IP. Print ONLY the IP and exit the script.
                echo "$LAN_IP"
                exit 0
            fi
        fi
    fi
done

# If the loop finishes without finding a valid IP
echo "Error: No active wired LAN interface found with an IP address." >&2
exit 1

