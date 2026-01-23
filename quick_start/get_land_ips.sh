#sudo apt install ssh nmap

NETWORK=$(echo $1 | cut -d. -f1-3).1/24

#echo $NETWORK

nmap $NETWORK | grep -v $1 | grep 'Nmap scan report for' | tr -cd '0-9.\n' > land_nodes_IP.txt    

tail -n +2 land_nodes_IP.txt > tmp_out && mv tmp_out land_nodes_IP.txt

echo "*********Node's found on Land network*********"
cat land_nodes_IP.txt

NUMBER_OF_NODES=$(wc -l < land_nodes_IP.txt)

echo "number of node's: "$NUMBER_OF_NODES

# Only generate key if it doesn't exist
if [ ! -f ~/.ssh/id_rsa ]; then
    ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
fi

for line in $(cat land_nodes_IP.txt); do
    if ssh $line "hostname" 2>/dev/null; then
        echo "SSH already working to $line"
    else
        ssh-copy-id $line
    fi
done