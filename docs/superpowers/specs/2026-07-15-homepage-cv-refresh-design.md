# Personal Website CV Refresh Design

## Goal

Update the public content of `quixoteh.github.io` to match Hai Huang's latest CV while retaining the existing AcademicPages theme and the verified Kaggle award.

## Non-negotiable content

The existing Biography paragraph in `_pages/about.md` must remain byte-for-byte unchanged:

> I recently completed my B.Eng. in Internet of Things Engineering at Northeast Agricultural University. I will begin my M.Sc. studies in Robotics and Intelligent Systems at Nanyang Technological University in Fall 2026. Previously, I served as an AI Solutions Engineer intern at China Unicom Chengdu Branch, Digital Technology Center. My research interests focus on computer vision and robotics.

The April 2024 Kaggle Home Credit silver medal remains on the site because it is a real achievement intentionally omitted from the targeted CV.

## Content changes

### Global profile and navigation

- Keep the current visual theme and page layout.
- Update the repository metadata and sidebar identity to Hai Huang's current status as an incoming M.Sc. student in Robotics and Intelligent Systems at Nanyang Technological University.
- Retain the GitHub and LinkedIn links and current profile image.
- Expose only relevant primary navigation: Research, Publications, Awards, and CV. Remove the empty or unrelated Blogs entry.

### Home page

- Preserve the Biography paragraph exactly.
- Replace interests with the CV focus: Robot Manipulation, Reinforcement Learning, Vision-Language-Action Models, and Computer Vision.
- Update education names, dates, coursework, and GPA from the CV.
- Update News with NTU admission, MSM-Seg status, SO-101, MuJoCo Playground, Marso Hack Berlin, China Unicom internship, and the retained Kaggle award.

### Research

Replace unrelated or generic portfolio material with concise project entries derived only from the CV:

1. SO-101 Real-World Robotic Learning with LeRobot.
2. MuJoCo Playground vision-based reinforcement learning for Panda grasping.
3. Marso Hack Berlin 2026 robot parcel sorting.
4. MSM-Seg multi-modal brain tumor segmentation research.
5. YOLOv8 and IoT agricultural pest detection undergraduate thesis.

Each entry will use the CV wording and dates without inventing results, links, or images.

### Publications

- Remove placeholder imagery, buttons, and template entries.
- Publish the single CV-listed MSM-Seg manuscript with the exact author/title/status information from the CV.
- Link the arXiv identifier `2510.10679` to its canonical arXiv abstract page.

### Awards

- Remove the placeholder image and button.
- Retain the April 2024 Kaggle Home Credit silver medal, including its top-0.5-percent ranking.

### CV

- Replace `_data/cv.json` template content with the latest CV's contact details, education, research interests, publication, projects, work experience, research experience, skills, and languages.
- Enable the CV navigation entry and render the JSON CV page.
- Do not expose the telephone number on the public website; the email remains public because it is already shown in the sidebar. The original `.tex` remains the private source of truth.

## Files and scope

Expected content changes are limited to `_config.yml`, `_data/navigation.yml`, `_data/cv.json`, `_pages/about.md`, `_pages/publications.html`, `_pages/portfolio.html`, `_pages/teaching.html`, `_pages/cv-json.md`, and the relevant `_publications`, `_portfolio`, and `_teaching` entries. Theme code and unrelated archived template examples will not be refactored.

## Verification

- Compare the protected Biography paragraph before and after with an exact string check.
- Parse `_data/cv.json` as JSON.
- Build the Jekyll site locally with Bundler.
- Scan generated pages for template placeholders and verify Research, Publications, Awards, and CV content.
- Inspect the generated site in a browser at desktop and mobile widths.
- Review `git diff` to ensure every changed line maps to this design.
