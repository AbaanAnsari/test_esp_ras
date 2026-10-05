# Raspberry Pi 5 quick start

```bash
cd ~/smart_hearing_aid_4mic/pi
chmod +x install.sh run.sh
./install.sh
./run.sh
```

Then open:

    http://127.0.0.1:8080

or from another computer:

    http://RASPBERRY_PI_IP:8080

To find the Pi IP:

    hostname -I

The UDP listener is port 5000.

If Linux firewall is enabled:

    sudo ufw allow 5000/udp
    sudo ufw allow 8080/tcp
