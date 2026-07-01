# Initialize a local Git repository
git init

# Add all your project files
git add .

# Commit your changes locally
git commit -m "Initial commit: Add student dataset and ML training script"

# Rename your default branch to main
git branch -M main

# Link your local repository to the GitHub repository you just created
# (Replace USERNAME and REPO-NAME with your actual GitHub details)
git remote add origin https://github.com

# Push your code to GitHub
git push -u origin main
