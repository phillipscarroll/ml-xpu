# Setting Up ARC GPUs with Full Mixed Precision in PyTorch on WSL2 (Windows)

## Step 1: Install WSL2
1. Open PowerShell as Administrator.
   ```bash
   wsl --install
   ```

## Step 2: Bridge WSL2 to the Network Interface Controller (NIC) for Native Access
1. Open "Windows Features" and enable **Hyper-V**.
2. Click **OK** and reboot your machine.
3. After rebooting, open **Hyper-V Manager** and set up a virtual switch:
   - Right-click your PC name and select **Virtual Switch Manager**.
   - Select **New Virtual Network Switch** > **External** > **Create Virtual Switch**.
   - Name the switch something memorable, like `wsl-nic`.
   - Set the external network to your LAN or WAN interface used for network connection.
   - Click **Apply** and **OK**.

4. Configure the WSL2 networking mode:
   - Open `%UserProfile%` in Windows Explorer.
   - Create a file named `.wslconfig` (include the leading dot).
   - Add the following lines to `.wslconfig`:
     ```
     [wsl2]
     networkingMode=bridged
     vmSwitch=wsl-nic
     ```

## Step 3: Install Ubuntu 22.04 on WSL2
1. Open PowerShell after the reboot and run:
   ```bash
   wsl --install --d Ubuntu-22.04
   ```
2. Set up your username and password.
   - **Note**: To remove this WSL2 instance, use:
     ```bash
     wsl --unregister Ubuntu-22.04
     ```
   - Restarting with a clean instance is often faster than troubleshooting issues.

## Step 4: Install Intel AI Tools
1. Download the Intel AI tools installer and run it:
   ```bash
   wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/491d5c2a-67fe-48d0-884f-6aecd88f5d8a/ai-tools-2025.0.0.75_offline.sh
   sh ai-tools-2025.0.0.75_offline.sh
   ```
   - Answer **yes** when prompted at the end.
   - Activate the Intel Python environment:
     ```bash
     source $HOME/intel/oneapi/intelpython/bin/activate
     ```

## Step 5: Configure Intel GPU Drivers
1. Install the Intel graphics GPG public key:
   ```bash
   wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
   ```

2. Configure the Intel package repository:
   ```bash
   echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
   ```

3. Update the package repository metadata:
   ```bash
   sudo apt update
   ```

4. Install compute-related packages:
   ```bash
   sudo apt-get install -y libze1 intel-level-zero-gpu intel-opencl-icd clinfo libze-dev intel-ocloc intel-level-zero-gpu-raytracing
   ```

5. Verify installation:
   ```bash
   clinfo | grep "Device Name"
   ```
   - You should see one or more Intel GPU instances listed.

## Step 6: Set Up JupyterLab with PyTorch
1. Create a directory for JupyterLab:
   ```bash
   mkdir ~/jupyterlabxpu
   cd ~/jupyterlabxpu
   ```

2. Activate your PyTorch GPU environment:
   ```bash
   conda activate pytorch-gpu
   ```

3. Install JupyterLab and related packages:
   ```bash
   pip install jupyterlab accelerate diffusers tqdm IProgress transformers scikit-learn
   ```

4. Set a JupyterLab password:
   ```bash
   jupyter lab password
   ```
   - Enter and confirm your password.

5. Start JupyterLab:
   - To run locally:
     ```bash
     jupyter lab
     ```
   - To run as a server:
     ```bash
     ip a
     jupyter lab --ip <WSL2_IP_ADDRESS> --port 8888
     ```
     - Replace `<WSL2_IP_ADDRESS>` with your WSL2 instance’s IP (e.g., `10.0.0.21`).

## Step 7: Restart the WSL Instance (if needed)
1. Shut down and restart your WSL instance:
   ```bash
   wsl --shutdown Ubuntu-22.04
   wsl
   ```

