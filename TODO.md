# FactoryGuard AI - Real-time Auth Enhancement TODO

## Approved Plan Steps (breakdown):

### Step 1: Dependencies & Setup [ ]
- Update package.json with socket.io-client, jsonwebtoken, bcryptjs, jwt-decode, react-router-dom
- Update backend/requirements.txt with flask-socketio, pyjwt, passlib[bcrypt]
- Run `npm install` && `cd backend && pip install -r requirements.txt -U`

### Step 2: Backend Authentication & Real-time [ ]
- Edit backend/app.py: Add JWT /login /register endpoints
- Add Flask-SocketIO, protect /predict with token
- Add '/factory' namespace: emit sensor_updates, predictions (simulate data)

### Step 3: Frontend Auth UI & Context [ ]
- Create src/context/AuthContext.tsx (Provider, login/logout, token mgmt)
- Create src/components/LoginModal.tsx, RegisterModal.tsx (Tailwind forms)
- Update src/main.tsx: Wrap App with AuthProvider

### Step 4: Frontend Real-time & Integration [ ]
- Edit src/App.tsx: Auth check (redirect/overlay), socket.io connect/listen updates
- Replace local predict → API /predict with auth header
- Add useSocket hook

### Step 5: Polish & Commits [ ]
- Error handling, logout, README updates
- Git branch blackboxai/feat-realtime-auth
- 5 commits: deps, backend-auth, frontend-auth, realtime-frontend, polish
- Test: npm dev + python backend/app.py

## Progress Tracking
- [x] Step 1 Complete
- [x] Step 2 Complete (Backend app.py with auth/WS/mock)
- [x] Step 3 Complete (Auth context/modals, main.tsx Provider)
- [ ] Step 4 Frontend integration/realtime
- [ ] Step 5 Commits/polish



