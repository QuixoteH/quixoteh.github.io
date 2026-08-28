# Profile Email And Experience Logos Design

## Goal

Add the missing email contact icon and official organization logos to the current homepage without changing any visible Experience wording or unrelated content.

## Approved Scope

- Add an icon-only email link beside the existing GitHub and LinkedIn profile links.
- The email target is exactly `mailto:quixotehh@gmail.com`.
- The email control has the accessible name and hover title `Email`.
- Change the homepage author heading from `Hai HUANG` to `Hai Huang`.
- Shorten the homepage profile title to exactly `M.S. Student in Robotics`.
- Rename the homepage biography section heading from `About` to `Biography` while keeping the navigation label `About`.
- Add the official A*STAR logo to `A*STAR Research Attachment`.
- Add the official China Unicom logo to `China Unicom AI Solutions Engineer Intern`.
- Keep both Experience titles exactly as written.
- Remove `GPA: 85/100` from the homepage Education card while retaining it on the CV page.

## Content And Data Model

The email address is stored in the existing `[social]` configuration and consumed by the existing profile social-link list. It does not appear as an additional text row.

Each Experience item gains only an `image` field pointing to a local asset under `public/logos/`. The entries must contain exactly `title` and `image`; no subtitle, institution, date, content, tags, dissertation wording, or SIMTech wording is added.

## Assets And Rendering

Use logo artwork published by the corresponding organization. Preserve the official artwork rather than redrawing, recoloring, or generating a substitute. Store local copies so the homepage does not depend on third-party image availability.

The existing `CardPage` image path and dimensions are reused for Experience. No new card component or layout variant is introduced. Logo assets must remain identifiable at the existing desktop and mobile card sizes without stretching or clipping.

The email icon uses Lucide's existing `Mail` icon so it matches the current GitHub and LinkedIn controls. The external-link attributes used by web profiles are omitted for the `mailto:` target when they are not applicable.

## Verification

Implementation is complete only when:

1. Content tests require the exact email address and exactly `title` plus `image` on both Experience entries.
2. Export checks require the rendered `mailto:` link and both local logo sources.
3. The full test suite, lint, production build, and static-export verification pass.
4. Desktop (`1280x720`) and mobile (`390x844`) browser checks show all three social icons, both Experience logos, no title wrapping regression, and no overflow or overlap.
5. The live GitHub Pages deployment succeeds and the public homepage matches the verified local result.

## Scope Boundaries

- Do not change Biography, News, Research, Publications, Awards, navigation, theme, or accent colors.
- Apart from the section heading, do not change the biography text.
- Apart from removing the homepage GPA line, do not change Education content.
- Keep the CV GPA unchanged.
- Do not change the CV name or other CV content.
- Do not change the profile institution line.
- Do not change either Experience title.
- Do not add visible email text, Experience metadata, or new homepage sections.
- Do not refactor unrelated profile or card behavior.
