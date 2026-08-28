# Experience Card Details Design

## Goal

Make the homepage Experience section use the same information hierarchy as Zhe Li's GentleFress cards, replace the blurry A*STAR asset, and give the site a serious academic favicon.

## Reference Structure

Reuse the existing PRISM `CardPage` fields in the same order as Zhe Li:

1. role title;
2. organization subtitle;
3. date badge;
4. location line; and
5. two compact keyword tags.

No new component or Experience-specific data model is needed.

## Exact Content

### A*STAR

- Title: `Research Attachment`
- Subtitle: `Singapore Institute of Manufacturing Technology (SIMTech), A*STAR`
- Date: `08/2026 – Present`
- Content: `Singapore`
- Tags: `Force-aware Robot Learning`, `Robot Manipulation`
- Image: `/logos/astar.png`

### China Unicom

- Title: `AI Solutions Engineer Intern`
- Subtitle: `China Unicom Chengdu Branch, Digital Technology Center`
- Date: `07/2025 – 09/2025`
- Content: `Chengdu, China`
- Tags: `AI Solution`, `IoT Solution`
- Image: `/logos/china-unicom.png`

The China Unicom role, organization, location, date, and keywords are derived from the latest CV source. The A*STAR institution, start date, current status, and research domain follow the user's confirmed attachment information and current research interests without adding supervisor or project-success claims.

## Logo

Replace the low-resolution A*STAR favicon with the exact official image supplied by the user:

`https://research.a-star.edu.sg/wp-content/uploads/2021/09/astar-vertical-logo-rgb-telegram-profile-pic.png`

Store it locally as `public/logos/astar.png` without redrawing or recoloring it.

## Site Icon

Replace the heart favicon with a simple open-book icon based on Lucide `BookOpen`. Use a Morandi purple `#7D6B8C` square background, white book strokes, no text, and the new path `/favicon-book.svg` so browsers do not reuse the old cached icon. Remove the obsolete heart asset.

## Rendering

The populated Experience entries use the standard card layout already used by Education and by Zhe Li's Experience section. Remove the earlier title-only compact branch because neither Experience item is title-only after this change. Desktop cards show the date badge at the upper right; narrow screens stack the date without horizontal overflow.

## Scope Boundaries

- Apart from the favicon path, do not change Biography, Education, News, Research, Publications, Awards, CV, navigation, profile, theme, or colors.
- Do not add supervisor names, detailed bullet points, or unverified A*STAR outcomes.
- Do not modify the China Unicom logo.

## Verification

- Content tests require the exact two objects and their field order.
- The A*STAR image must be a `675 x 675` PNG matching the supplied official asset.
- The exported metadata must reference `/favicon-book.svg`, whose accessible title is `Open book`.
- Tests, lint, build, and static export must pass.
- Desktop and mobile screenshots must match the Zhe Li hierarchy, load both logos, and have no overflow or overlap.
- GitHub Pages deployment and the uncached live page must be verified.
