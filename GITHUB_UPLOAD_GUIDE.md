# GitHub Upload Guide

Step-by-step guide to upload your project to GitHub so you can work on it from another device.

## Prerequisites

1. **Git installed** ✅ (You have git version 2.50.0)
2. **GitHub account** (Create one at https://github.com if you don't have one)
3. **GitHub repository** (Create one on GitHub.com first)

## Step-by-Step Instructions

### Step 1: Create GitHub Repository

1. Go to https://github.com and sign in
2. Click the **"+"** icon in the top right → **"New repository"**
3. Fill in:
   - **Repository name**: `traffic-signal-control` (or your preferred name)
   - **Description**: "Traffic Signal Control using DQN and SUMO Simulation"
   - **Visibility**: Choose **Private** (for FYP) or **Public**
   - **DO NOT** check "Initialize with README" (we already have files)
4. Click **"Create repository"**
5. **Copy the repository URL** (e.g., `https://github.com/yourusername/traffic-signal-control.git`)

### Step 2: Initialize Git Repository (Local)

Open PowerShell in your project folder (`E:\FYP`) and run:

```powershell
# Navigate to project folder
cd E:\FYP

# Initialize git repository
git init

# Configure your git identity (if not already done)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Verify .gitignore exists (already created for you)
```

### Step 3: Add All Files

```powershell
# Check what will be added (optional - preview)
git status

# Add all files (respects .gitignore)
git add .

# Verify what's staged (optional)
git status
```

**Note**: The `.gitignore` file I created will automatically exclude:
- Python cache files (`__pycache__/`)
- Output files (`output/`, `*.csv`, `*.json`)
- Large datasets (`TAPASCologne-0.32.0/`, `setup/*.msi`)
- SUMO log files (`*.log`, `sumo.log`)
- Model files (`*.pth`, `*.pt`)
- Temporary files

### Step 4: Commit Files

```powershell
# Create initial commit
git commit -m "Initial commit: Traffic Signal Control System

- SUMO traffic simulation
- DQN, Adaptive, and Fixed-time controllers
- Flask web interface
- Academic evaluation tools
- Single and 4-intersection networks"
```

### Step 5: Connect to GitHub Repository

```powershell
# Add remote repository (replace URL with your repository URL)
git remote add origin https://github.com/yourusername/your-repo-name.git

# Verify remote was added
git remote -v
```

**Important**: Replace `yourusername/your-repo-name` with your actual GitHub username and repository name!

### Step 6: Push to GitHub

```powershell
# Push to GitHub (first time)
git branch -M main
git push -u origin main
```

**Note**: 
- If prompted, enter your GitHub username and password
- For password: Use a **Personal Access Token** (PAT) instead of password
  - GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
  - Generate new token with `repo` scope
  - Use the token as password

## Complete Command Sequence

Here's the complete sequence (copy-paste friendly):

```powershell
cd E:\FYP

# Initialize
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add files
git add .

# Commit
git commit -m "Initial commit: Traffic Signal Control System"

# Connect to GitHub (REPLACE WITH YOUR REPO URL!)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push
git branch -M main
git push -u origin main
```

## Alternative: Using GitHub Desktop (Easier)

If you prefer a GUI:

1. Download **GitHub Desktop** from https://desktop.github.com/
2. Install and sign in with your GitHub account
3. Click **"File" → "Add local repository"**
4. Select `E:\FYP`
5. Click **"Publish repository"**
6. Choose repository name and visibility
7. Click **"Publish repository"**

## Files Excluded by .gitignore

The following will **NOT** be uploaded (to save space):

- ✅ **Python cache**: `__pycache__/`, `*.pyc`
- ✅ **Output files**: `output/*.csv`, `output/*.json`
- ✅ **Large datasets**: `TAPASCologne-0.32.0/`, `TC-DQN-master/`
- ✅ **SUMO logs**: `*.log`, `sumo.log`
- ✅ **Model files**: `*.pth`, `*.pt` (in `models/`)
- ✅ **Temporary files**: `temp_*.py`, `temp_*.json`
- ✅ **Installer files**: `setup/*.msi`
- ✅ **TensorBoard logs**: `runs/`, `logs/`

**These WILL be uploaded**:

- ✅ Source code: `src/*.py`
- ✅ Configuration: `*.xml` (network files), `*.sumocfg`, `*.rou.xml`
- ✅ Templates: `templates/*.html`
- ✅ Documentation: `*.md` files
- ✅ Requirements: `requirements.txt`
- ✅ Flask app: `app_flask.py`, `app.py`

## On Your Other Device

After uploading, on your other device:

```powershell
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Navigate to project
cd YOUR_REPO_NAME

# Install dependencies
pip install -r requirements.txt

# Download SUMO separately (if needed)
# The SUMO installer is excluded from git (too large)
```

## Future Updates (Working on Another Device)

When you make changes on another device:

```powershell
# Check status
git status

# Add changes
git add .

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push
```

## Pulling Changes (Syncing)

When you switch devices, pull the latest changes:

```powershell
# Pull latest changes
git pull

# Or if you're on a different branch
git pull origin main
```

## Troubleshooting

### Error: "Repository not found"
- Check your repository URL is correct
- Verify you're logged in to GitHub
- Check repository visibility (private repos need authentication)

### Error: "Authentication failed"
- Use Personal Access Token instead of password
- GitHub → Settings → Developer settings → Personal access tokens
- Generate token with `repo` scope

### Error: "Large files"
- If you get errors about large files, check `.gitignore` is working
- Remove large files: `git rm --cached large_file.txt`
- The `.gitignore` I created should handle this automatically

### Error: "Already exists"
- If remote already exists: `git remote set-url origin NEW_URL`

## Recommended Repository Structure on GitHub

Your repository structure should look like:

```
traffic-signal-control/
├── src/                    # Source code
├── Dataset/                # Network files (configs only)
├── templates/              # Flask templates
├── static/                 # Static files
├── requirements.txt        # Python dependencies
├── app_flask.py           # Flask application
├── README.md              # Documentation
├── .gitignore             # Git ignore rules
└── [other config files]
```

## Additional Tips

1. **Regular Commits**: Commit often with clear messages
2. **Branching**: Use branches for features (`git checkout -b feature-name`)
3. **README**: Update README.md with setup instructions
4. **Large Files**: Use Git LFS for very large files (if needed)
5. **Private Repo**: Keep FYP private until submission

## Quick Reference Commands

```powershell
# Check status
git status

# Add all changes
git add .

# Commit
git commit -m "Your commit message"

# Push to GitHub
git push

# Pull from GitHub
git pull

# Check remote URL
git remote -v

# View commit history
git log
```

## Need Help?

- **Git Documentation**: https://git-scm.com/doc
- **GitHub Docs**: https://docs.github.com
- **Git Cheat Sheet**: https://education.github.com/git-cheat-sheet-education.pdf

