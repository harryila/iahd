# Logit Lens on GPT-2 Activations

This project implements the "logit lens" technique for interpreting GPT-2's internal activations.

- [Original notebook](https://colab.research.google.com/drive/1-nOE-Qyia3ElM17qrdoHAtGmLCPUZijg)
- [Blog post explanation](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

---

## Team Setup Guide

We use Lambda Labs GPU instances for running this code. Eric's API key handles billing.

### Step 1: Get your SSH public key

```bash
# Check if you have one
cat ~/.ssh/id_ed25519.pub
# or
cat ~/.ssh/id_rsa.pub

# If not, generate one:
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Copy the output (starts with `ssh-ed25519` or `ssh-rsa`).

### Step 2: Add your SSH key to the Lambda account

Get the API key from Eric (ask in Slack/Discord). Then run:

```bash
export LAMBDA_API_KEY='<paste-api-key-here>'

python3 -c "
import requests
API_KEY = '$LAMBDA_API_KEY'
YOUR_NAME = 'your-name-here'  # Change this!
YOUR_KEY = '''$(cat ~/.ssh/id_ed25519.pub)'''  # Or id_rsa.pub

r = requests.post('https://cloud.lambdalabs.com/api/v1/ssh-keys',
    headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'},
    json={'name': YOUR_NAME, 'public_key': YOUR_KEY})
print('Success!' if r.status_code == 200 else f'Error: {r.text}')
"
```

### Step 3: Launch a GPU instance

```bash
export LAMBDA_API_KEY='<paste-api-key-here>'

python3 -c "
import requests, json
API_KEY = '$LAMBDA_API_KEY'
YOUR_NAME = 'your-name-here'  # Same name you used above

# Launch an A10 instance (\$0.75/hr)
r = requests.post('https://cloud.lambdalabs.com/api/v1/instance-operations/launch',
    headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'},
    json={
        'region_name': 'us-east-1',
        'instance_type_name': 'gpu_1x_a10',
        'ssh_key_names': [YOUR_NAME]
    })
print(json.dumps(r.json(), indent=2))
"
```

### Step 4: Get instance IP address

```bash
python3 -c "
import requests, json
API_KEY = '$LAMBDA_API_KEY'
r = requests.get('https://cloud.lambdalabs.com/api/v1/instances',
    headers={'Authorization': f'Bearer {API_KEY}'})
for i in r.json().get('data', []):
    print(f\"{i['id']}: {i['ip']} ({i['status']})\")
"
```

### Step 5: Connect via Cursor

1. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
2. Type: `Remote-SSH: Connect to Host...`
3. Enter: `ssh ubuntu@<instance-ip>`
4. Select the host to connect

### Step 6: Set up the instance (first time only)

Once connected via SSH:

```bash
# Verify GPU
nvidia-smi

# Clone this repo
git clone https://github.com/harryila/iahd.git
cd iahd

# Install dependencies
pip install tensorflow==1.15 colorcet matplotlib pandas numpy scipy

# Clone GPT-2 and download model
git clone https://github.com/openai/gpt-2.git
cd gpt-2
python download_model.py 1558M
```

### Step 7: Stop the instance when done

**Important: Instances cost money! Stop them when not using.**

```bash
python3 -c "
import requests
API_KEY = '$LAMBDA_API_KEY'
INSTANCE_ID = 'paste-instance-id-here'

r = requests.post('https://cloud.lambdalabs.com/api/v1/instance-operations/terminate',
    headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'},
    json={'instance_ids': [INSTANCE_ID]})
print('Terminated!' if r.status_code == 200 else f'Error: {r.text}')
"
```

---

## Team Collaboration

### Option A: Each person gets their own instance
- Each team member launches their own instance
- Work independently, share code via Git
- Best for parallel work

### Option B: Share one instance
- One person launches the instance
- Add all team members' SSH keys when launching:
  ```python
  'ssh_key_names': ['harry', 'teammate1', 'teammate2']
  ```
- Everyone SSHs to the same instance
- Coordinate who runs code (one GPU = one job at a time)

### Git workflow
```bash
# Pull latest before starting
git pull

# Make changes, then push
git add .
git commit -m "Your changes"
git push
```

---

## Instance Types & Pricing

| Type | GPU | Price/hr |
|------|-----|----------|
| gpu_1x_a10 | 1x A10 (24GB) | $0.75 |
| gpu_1x_a100_sxm4 | 1x A100 (40GB) | $1.29 |
| gpu_1x_h100_pcie | 1x H100 (80GB) | $2.49 |
