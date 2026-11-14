# Publish this major project to GitHub

Your major project must be in its own **public** repository. Do not upload it into the existing minor-project repository.

## 1. Create the empty repository

Open https://github.com/new and set:

- Repository name: `Smart-Fake-News-Detection-System`
- Visibility: **Public**
- Do not initialize it with a README, `.gitignore`, or license (this bundle already contains those files).

## 2. Upload the project

Extract `Smart_Fake_News_Detection.zip`. Open the new repository, choose **Add file → Upload files**, drag the extracted project contents (not the ZIP itself), and commit the files to `main`.

## 3. Or push with Git

From a terminal opened inside the extracted project folder, run:

```bash
git init -b main
git add .
git commit -m "Build Smart Fake News Detection System major project"
git remote add origin https://github.com/shivam-pandyacoder24/Smart-Fake-News-Detection-System.git
git push -u origin main
```

## 4. Finish the evidence before submission

Download the dataset, run preparation and training, verify the generated results, and commit the generated `models/`, `outputs/` and updated `Project_Report.pdf`. Capture real application screenshots, record the demonstration video and deploy Streamlit if you are including a live link. See `PROJECT_CHECKLIST.md`.
