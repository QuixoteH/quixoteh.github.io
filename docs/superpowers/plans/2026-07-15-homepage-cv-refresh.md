# Personal Website CV Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh every public profile page from Hai Huang's latest CV while preserving the approved Biography paragraph and Kaggle award.

**Architecture:** Keep the AcademicPages/Jekyll theme. Update source Markdown, YAML, and JSON; use existing collections for Research, Publications, and Awards, then verify the generated static site.

**Tech Stack:** Jekyll, Liquid, Markdown, YAML, JSON, Bundler, Python 3, GitHub Pages.

## Global Constraints

- Preserve the existing Biography paragraph in `_pages/about.md` byte-for-byte.
- Preserve the April 2024 Kaggle Home Credit silver medal and top-0.5-percent ranking.
- Use only facts in the supplied CV, except for the approved Kaggle award.
- Keep the theme, profile image, GitHub link, and LinkedIn link.
- Do not publish the telephone number.
- Do not refactor theme code or unrelated archived examples.

---

### Task 1: Global Profile, Navigation, and Home Page

**Files:**
- Modify: `_config.yml`
- Modify: `_data/navigation.yml`
- Modify: `_pages/about.md`

**Interfaces:**
- Consumes: Existing AcademicPages configuration and approved design.
- Produces: Current sidebar identity, navigation, interests, education, and news.

- [ ] **Step 1: Run the pre-change contract and confirm failure**

```bash
python3 -c "from pathlib import Path; c=Path('_config.yml').read_text(); a=Path('_pages/about.md').read_text(); assert 'repository               : \"QuixoteH/quixoteh.github.io\"' in c; assert 'Robot Manipulation' in a; assert '[Jul 2026] Reproduced Google DeepMind' in a"
```

Expected: `AssertionError` because metadata and home content are stale.

- [ ] **Step 2: Update global content**

Set `_config.yml` to repository `QuixoteH/quixoteh.github.io`; sidebar bio `Incoming M.Sc. student in Robotics and Intelligent Systems at NTU`; location `Singapore`; employer `Nanyang Technological University`. Keep current profile image, GitHub, and LinkedIn values. Replace visible navigation with Research, Publications, Awards, and CV (`/cv-json/`).

- [ ] **Step 3: Update the home page around the protected Biography**

Use these interests:

```markdown
- Robot Manipulation
- Reinforcement Learning
- Vision-Language-Action Models
- Computer Vision
```

Use the two CV education records, dates, GPA, and coursework. Write News entries for MSM-Seg, Marso Hack Berlin, MuJoCo Playground, SO-101, NTU admission, China Unicom, and the retained Kaggle medal.

- [ ] **Step 4: Verify home content**

```bash
python3 -c "from pathlib import Path; a=Path('_pages/about.md').read_text(); p='I recently completed my B.Eng. in Internet of Things Engineering at Northeast Agricultural University. I will begin my M.Sc. studies in Robotics and Intelligent Systems at Nanyang Technological University in Fall 2026. Previously, I served as an AI Solutions Engineer intern at China Unicom Chengdu Branch, Digital Technology Center. My research interests focus on computer vision and robotics.'; assert a.count(p)==1; assert 'Robot Manipulation' in a; assert 'Kaggle Competition: Home Credit' in a"
```

Expected: exit 0.

---

### Task 2: Research, Publications, and Awards

**Files:**
- Modify: `_pages/portfolio.html`, `_pages/publications.html`, `_pages/teaching.html`
- Delete: `_portfolio/portfolio-1.json`, `_portfolio/portfolio-2.json`, `_publications/md1.md`, `_teaching/md2.md`
- Create: `_portfolio/2025-msm-seg.md`, `_portfolio/2026-so101.md`, `_portfolio/2026-mujoco-playground.md`, `_portfolio/2026-marso.md`, `_portfolio/2026-pest-detection.md`
- Create: `_publications/2025-msm-seg.md`
- Create: `_teaching/2024-kaggle-home-credit.md`

**Interfaces:**
- Consumes: Existing Jekyll collection rendering and CV facts.
- Produces: Focused public Research, Publications, and Awards pages.

- [ ] **Step 1: Run the collection contract and confirm failure**

```bash
python3 -c "from pathlib import Path; assert len(list(Path('_portfolio').glob('*.md')))==5; assert Path('_publications/2025-msm-seg.md').exists(); assert Path('_teaching/2024-kaggle-home-credit.md').exists()"
```

Expected: `AssertionError` because the entries do not exist.

- [ ] **Step 2: Simplify landing pages**

Retain front matter, `{% include base_path %}`, and collection loops. Remove unrelated images, `666` buttons, and duplicate images. Rename the Research page title to `Research & Projects`.

- [ ] **Step 3: Create five Research records**

Each record has front matter containing `title`, `excerpt`, `collection: portfolio`, and the CV start date, followed by its exact CV bullet points. The five titles are SO-101 Real-World Robotic Learning with LeRobot; MuJoCo Playground -- Vision-Based RL for Panda Arm Grasping; Marso Hack Berlin 2026 -- Robot Parcel Sorting Challenge; MSM-Seg: Multi-Modal Brain Tumor Segmentation; and the YOLOv8/IoT agricultural pest detection thesis. Do not add unverified results, links, or images.

- [ ] **Step 4: Create the publication record**

Use category `manuscripts`, date `2025-10-14`, venue `arXiv`, paper URL `https://arxiv.org/abs/2510.10679`, the exact author/title string from the CV, and status `under review at IEEE TMM after major revision`.

- [ ] **Step 5: Create the retained award record**

Use title `Silver Medal -- Home Credit: Credit Risk Model Stability`, collection `teaching`, type `Kaggle Competition Award`, venue `Kaggle`, date `2024-04-01`, and excerpt `Silver medal, ranking in the top 0.5% of teams.`

- [ ] **Step 6: Verify collection content**

```bash
python3 -c "from pathlib import Path; files=list(Path('_portfolio').glob('*.md')); text='\n'.join(p.read_text() for p in files); assert len(files)==5; assert all(x in text for x in ['SO-101','MuJoCo Playground','Marso Hack Berlin','MSM-Seg','YOLOv8']); assert 'quantitative finance' not in text.lower(); assert '666' not in ''.join(Path(p).read_text() for p in ['_pages/portfolio.html','_pages/publications.html','_pages/teaching.html'])"
```

Expected: exit 0.

---

### Task 3: Structured CV Page

**Files:**
- Modify: `_data/cv.json`
- Modify: `_pages/cv-json.md`

**Interfaces:**
- Consumes: Fields supported by `_includes/cv-template.html`.
- Produces: Public CV at `/cv-json/` without phone or broken PDF link.

- [ ] **Step 1: Run the CV contract and confirm failure**

```bash
python3 -c "import json; d=json.load(open('_data/cv.json')); assert d['basics']['name']=='Hai HUANG'; assert d['basics']['phone']==''; assert len(d['portfolio'])==5; assert d['publications'][0]['name'].startswith('MSM-Seg')"
```

Expected: `AssertionError` because template data remains.

- [ ] **Step 2: Replace CV JSON**

Populate `basics`, `work`, `education`, `skills`, `languages`, `interests`, `publications`, and `portfolio` from the supplied CV. Use empty arrays for `references`, `presentations`, and `teaching`. Set name `Hai HUANG`, label `Incoming M.Sc. Student in Robotics and Intelligent Systems`, email `quixotehh@gmail.com`, empty phone, website `https://quixoteh.github.io`, Singapore location, GitHub `QuixoteH`, and LinkedIn `quixoteh`. Include every CV bullet in its matching work or portfolio record.

- [ ] **Step 3: Remove the broken CV download control**

Keep the layout, stylesheets, and `{% include cv-template.html %}`. Remove the missing `/files/cv.pdf` link and retain a `Back to Home` link.

- [ ] **Step 4: Validate JSON and privacy constraints**

```bash
python3 -m json.tool _data/cv.json >/dev/null && python3 -c "import json; d=json.load(open('_data/cv.json')); assert d['basics']['name']=='Hai HUANG'; assert d['basics']['phone']==''; assert len(d['portfolio'])==5; assert len(d['work'])==1; assert '2510.10679' in d['publications'][0]['website']"
```

Expected: exit 0.

---

### Task 4: Build, Visual Check, Backup, and Delivery

**Files:**
- Modify only if verification exposes an in-scope content or rendering defect.
- Back up final documents to `/home/quixoteh/documents/学习相关/obsidian/HH/课业/科研/个人主页/`.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: A validated branch ready for GitHub.

- [ ] **Step 1: Install repository dependencies if absent**

Run `bundle check`; if missing, run `bundle install --path vendor/bundle`, then confirm `bundle check` reports satisfied dependencies.

- [ ] **Step 2: Build and verify generated content**

```bash
bundle exec jekyll build
python3 -c "from pathlib import Path; s='\n'.join(p.read_text(errors='ignore') for p in Path('_site').rglob('*.html')); required=['SO-101 Real-World Robotic Learning','MuJoCo Playground','Marso Hack Berlin 2026','MSM-Seg','Home Credit: Credit Risk Model Stability','Hai HUANG']; assert all(x in s for x in required); forbidden=['Your Sidebar Name','GitHub University','Paper Title Number 1','AI for quantitative finance']; assert all(x not in s for x in forbidden)"
```

Expected: Jekyll exits 0 and the content check exits 0.

- [ ] **Step 3: Inspect desktop and mobile pages**

Serve on `127.0.0.1:4000`; inspect `/`, `/portfolio/`, `/publications/`, `/teaching/`, and `/cv-json/`. Verify navigation, wrapping, collection links, and absence of broken media or controls.

- [ ] **Step 4: Review, commit, back up, and push**

Run `git diff --check`, inspect `git status --short`, `git diff --stat`, and `git diff`, rerun the Biography exact-string test, commit with `feat: refresh website from latest CV`, copy the `.tex`, design, plan, and change summary into the Obsidian project folder, then push `codex/update-homepage-from-cv` after every check passes.
