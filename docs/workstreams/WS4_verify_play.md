# WS4 Verify and Play: verification, app and evaluation

**Owner:** Sarade Shreyash Jitendra · **Paths:** `apps/web/`, `supabase/`, `services/inference/app/verify.py`, `.github/`

**You produce:** the Next.js app on Vercel, Supabase schema and security, QR codes, verification, leaderboard, the field test.
**You consume:** the model API (generated types in `apps/web/src/contracts/`), policies (bin keys for QR codes).

## Suggested work

**Until 6 Oct**
- [ ] Create the GitHub repo, add everyone, fill in `CODEOWNERS`, set the `main` ruleset (README)
- [ ] Vercel project (root `apps/web`) and Supabase project; run the migration; enable anonymous sign-in
- [ ] Capture → result flow in the web app against the stub API on the Space (or dummy responses)

**7–17 Oct**
- [ ] Route handlers write scans to Supabase after `/predict`
- [ ] QR generation for demonstration bins (payload must pass `parseBinQr`) and in-browser scanning
- [ ] `/api/verify` route: load the original scan from Supabase, call the model API, write the disposal

**18–24 Oct**
- [ ] Leaderboard page from the `leaderboard` view (pseudonyms only)
- [ ] Verdict screen: bin, heatmap, rule trace, tip, NEA link; "waking up the model" state
- [ ] Vision-language-model baseline script on the same test photos

**25–31 Oct**
- [ ] Field test with at least 50 unseen items; costly-error rate, verification false-accept rate
- [ ] Demo check: wake the Space, run the full flow end to end

## Definition of done
- `pnpm check:web` and `pnpm --filter web build` pass; API shapes come only from `@/lib/contracts`
- No secret reaches the browser (HF token and service-role key are used only in route handlers)
- RLS on: a user can read only their own scans; the leaderboard exposes pseudonyms and points only
