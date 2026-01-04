# Sector Constraints Reminder

This file is a short reminder to implement full sector-constraint handling in the ML recommender.

Tasks to complete later:

- Replace penalty-based sector handling with hard constraints in the optimizer (e.g., via integer programming or constrained evolutionary algorithms).
- Add a validated symbol->sector mapping for the Nifty50 and Next50 asset universe.
- Implement logic to handle unknown symbols or missing sector mappings:
  - fallback: treat unknowns as a separate "Unknown" sector with conservative limits
  - option: require the uploader to provide full mapping before running optimization
- Add unit tests to verify sector exposures never exceed `max_sector_exposure` for recommended portfolios.
- Add UI to allow users to set per-sector limits and to lock specific holdings.

Reminder created: include this before running production recommendations.
