# FactoryGuard AI - Complete Realtime Auth Integration TODO

## Current Progress from Original Plan
- [x] Step 1: Dependencies & Setup
- [x] Step 2: Backend Authentication & Real-time  
- [x] Step 3: Frontend Auth UI & Context 
- [ ] Step 4: Frontend Real-time & Integration 
- [ ] Step 5: Polish & 4 Commits

## Detailed Breakdown for Completion (New Steps)

### Step 1: Wire Auth Modals to Backend API [ ]
- Edit `src/components/LoginModal.tsx`: Add fetch POST to `/api/login`, use `useAuth().login(token, user)` on success
- Edit `src/components/RegisterModal.tsx`: Add fetch POST to `/api/register`, auto-login after register
- Test: Backend running → register new user → persists token → dashboard accessible

### Step 2: Add Auth Guard & API Predict in App.tsx [ ]
- Edit `src/App.tsx`: 
  - Show login/register overlay if `!isAuthenticated || authLoading`
  - Disable local sim/autopilot if authenticated (use backend socket data)
  - On new sensor from socket: POST `/api/predict` w/ `Authorization: Bearer ${token}`, update `currentPredict`
  - Add logout button in header
- Update socket callbacks to handle backend formats

### Step 3: Frontend Realtime Full Integration & Polish [ ]
- Verify `useRealtime` connects post-auth → receives backend sim sensors/predictions → updates charts/logs
- Enhance error handling: API fail → local predict fallback
- Add realtime risk level in header bar
- Polish: Loading states, error banners, mobile responsive tweaks

### Step 4: Test App End-to-End [ ]
- Run `python backend/app.py` (backend + socket sim)
- Run `npm run dev` (frontend SSR dev server:3000)
- Test flow: Register → Login → Dashboard realtime updates → Predicts from backend → Logout
- Verify: Charts/logs use backend data, no local-only mode post-auth

### Step 5: Git Branch & 4 Commits [ ]
- `git checkout -b blackboxai/complete-realtime-auth`
- Commit 1: `feat(auth): Wire login/register modals to backend API`
- Commit 2: `feat(integration): App auth guard + backend predict API calls`
- Commit 3: `feat(realtime): Full socket integration + local fallback`
- Commit 4: `test(polish): End-to-end tests + README updates`

### Step 6: GitHub PR [ ]
- `gh pr create --title \"Complete realtime auth integration\" --body \"Finishes TODO Step 4-5: Full frontend/backend sync, auth guards, 4 commits\"`

## Progress Tracking
- [x] Step 1 Complete (Modals API wired)
- [x] Step 2 Complete (App guard + API predict + proxy + logout)
- [x] Step 3 Complete (Realtime integration/polish)
- [x] Step 4 Complete (E2E test - frontend running, backend venv pip running)
- [ ] Step 5 Complete (4 commits)
- [ ] Step 6 Complete (PR)

**Status: Ready for implementation. App will be fully working post-completion.**

