# GentleFress Homepage Refresh Design

## Goal

Refresh the current PRISM site using Zhe Li's GentleFress homepage as the structural and visual reference while preserving Hai Huang's verified content and existing routes.

The change must:

- make light mode the first-visit default;
- add `About` to the existing navigation;
- replace the homepage `Selected Publications` section with `Education` and `Experience`;
- restyle `News` to match the GentleFress list presentation;
- replace the current gold accent with a muted Morandi purple; and
- correct confirmed stale facts on the website CV without rewriting the CV.

## Sources Of Truth

- Current site baseline: `QuixoteH/quixoteh.github.io` commit `e1501b3` on `master`.
- Visual and structural reference: `gentlefress/gentlefress.github.io` commit `c9c75ef` and the deployed GentleFress homepage.
- CV terminology and dated facts: the latest provided CV source.
- Existing public biography and news remain authoritative unless an explicit change is listed below.

The GentleFress repository is a reference for layout, component behavior, and writing structure. Its personal facts, institutions, projects, links, emoji, and copy must not be imported.

## Information Architecture

The top navigation becomes:

| Label | URL |
| --- | --- |
| About | `/` |
| Research | `/portfolio/` |
| Publications | `/publications/` |
| Awards | `/teaching/` |
| CV | `/cv-json/` |

No `Services` page is added. All existing routes remain available.

## Homepage Structure

The existing two-column PRISM layout remains. On desktop the profile stays in the left column and the content stays in the right column. On mobile the columns stack without changing content order.

The left profile adds this exact line below the portrait and name:

`M.Sc. Student in Robotics and Intelligent Systems`

The right column follows the GentleFress sequence:

1. `About`
2. `Education`
3. `Experience`
4. `News`

`Selected Publications` is removed from the homepage only. The standalone Publications page and its content are unchanged.

### About

The current biography wording remains unchanged. Markdown links and emphasis may be added only as presentation metadata; they must not alter the visible wording or factual claims.

### Education

Education uses the same logo-card pattern as GentleFress. It contains only the two verified degrees from the latest CV:

- `M.S. in Robotics and Intelligent Systems`, Nanyang Technological University, `Aug. 2026 - Jan. 2028 (Expected)`, with School of Mechanical and Aerospace Engineering details.
- `B.E. in Internet of Things`, Northeast Agricultural University, `Sep. 2022 - Jun. 2026`, with College of Intelligent Science and Engineering details and the existing GPA information.

The degree abbreviations `M.S.` and `B.E.` are correct and must not be normalized to other abbreviations. Institution logos are presentation assets, not new content.

### Experience

Experience contains exactly two compact, single-line entries:

- `A*STAR Research Attachment`
- `China Unicom AI Solutions Engineer Intern`

Do not add a subtitle, institution line, date, description, tags, dissertation wording, or SIMTech wording to either entry. The single-line presentation is an explicit content constraint even though GentleFress cards can display more fields.

### News

The seven current news entries keep their existing wording, dates, and order. Presentation follows GentleFress: a bounded scrollable list, stable date column, readable body column, separators, and responsive stacking on narrow screens. Do not import GentleFress emoji or news wording.

## Theme And Color

Light mode is the default when no explicit user preference has been stored. The theme control remains available so a visitor can choose another mode.

The global gold accent is replaced with Morandi purple:

- light-theme accent: `#7D6B8C`;
- dark-theme accent: a lighter matching purple with sufficient contrast.

The accent applies in the same semantic places where GentleFress uses gold: links, the profile title, education institution text, and active or interactive highlights. Biography links and education subtitles use this accent role; arbitrary body words do not receive manual color spans. Primary body text and headings remain neutral rather than turning the page into a one-color theme.

## CV Corrections

The CV page receives only confirmed factual or temporal corrections:

- replace `Incoming M.Sc. Student in Robotics and Intelligent Systems` with `M.Sc. Student in Robotics and Intelligent Systems`;
- change the SO-101 project date to `2026-07-01`, matching the latest CV;
- replace the stale present-progress Marso summary with: `Built an imitation-learning pipeline for simulated Franka Panda parcel sorting in ManiSkill3, covering color-labeled parcel placement across three difficulty settings, State Diffusion Policy baselines, Easy RGB ACT offline training, and fixed 50-episode evaluation.`

The following are explicitly unchanged:

- `M.S.` and `B.E.` degree abbreviations;
- the MSM-Seg CV/publication status, which remains unchanged until the user decides to update it after publication;
- unrelated CV wording, skills, projects, and links.

## Implementation Approach

Use the mechanism visible in the GentleFress PRISM export rather than building a separate homepage system:

- extend homepage section loading to support PRISM card sections;
- store Education and Experience in focused TOML content files;
- reuse the existing `CardPage` component;
- add only the conditional logo layout needed by Education cards;
- use the existing `News` component with the GentleFress list styling;
- update navigation, profile title, theme defaults, and accent tokens in their existing owners.

Cards without images or extra metadata must retain their current behavior. The Research and Awards pages must not receive unrelated visual or content changes.

## Responsive And Accessibility Behavior

- Education cards keep logo, text, date, and tags within stable responsive tracks.
- Experience remains single-line and does not gain hidden metadata on mobile.
- News dates and text stack cleanly on narrow screens.
- Morandi purple text must meet readable contrast against its background.
- Keyboard navigation, visible focus behavior, semantic headings, and external-link behavior remain intact.
- Server-rendered homepage content must not contain hidden `opacity: 0` states.

## Verification And Deployment

Implementation is complete only when all of the following pass:

1. Content/config tests confirm navigation order, homepage section order, exact Experience strings, retained news, and CV corrections.
2. Lint and production build succeed.
3. Static export verification confirms all existing routes and rejects hidden SSR content.
4. A fresh-storage browser session opens in light mode.
5. Theme switching remains functional.
6. Desktop (`1280x720`) and mobile (`390x844`) screenshots show no overflow, overlap, missing logo, blank content, or broken navigation.
7. The homepage has no `Selected Publications` section.
8. The live GitHub Pages deployment succeeds and the uncached public URL matches the verified local build.

After local verification, push the focused commit to `master`, monitor GitHub Actions to completion, and verify `https://quixoteh.github.io/` directly.

## Scope Boundaries

- Do not add Services, Teaching, a blog, analytics, or other template sections.
- Do not import GentleFress personal content or emoji.
- Do not rewrite the biography or seven news items.
- Do not change the MSM-Seg CV/publication status in this task.
- Do not add unverified education, experience, supervisor, project, or research claims.
- Do not refactor unrelated PRISM components.
