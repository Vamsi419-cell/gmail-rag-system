import os
import sys
import json
import secrets
from functools import wraps
from pathlib import Path

from flask import Flask, redirect, url_for, session, request, jsonify, render_template_string, abort
from dotenv import load_dotenv
from google_auth_oauthlib.flow import Flow
import requests as http_requests

# ── Ensure src/ is importable ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

load_dotenv(BASE_DIR / ".env")

from ingestion import init_db, upsert_user, authenticate_gmail, backfill, delta_sync, get_user_email_count
from process import build_user_index
from rag_system import ask_my_emails, invalidate_user_cache

# ── Flask App ──────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))

# Secure session cookies
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

# Allow OAuth over HTTP for local dev
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# Google OAuth config
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]
REDIRECT_URI = "http://localhost:5000/callback"

# Initialize database on startup
init_db()


# ── Helpers ────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            if request.is_json:
                return jsonify({"status": "error", "message": "Session expired. Please refresh and log in again."}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def _check_origin():
    """Basic CSRF protection: reject cross-origin POST requests."""
    origin = request.headers.get("Origin", "")
    referer = request.headers.get("Referer", "")
    allowed = ("http://localhost:5000", "http://127.0.0.1:5000")
    if not (origin.startswith(allowed) if isinstance(origin, str) and origin else True):
        if not any(referer.startswith(a) for a in allowed):
            abort(403)


def _build_flow():
    client_config = {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [REDIRECT_URI],
        }
    }
    flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=REDIRECT_URI)
    return flow


# ═══════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.route("/login")
def login():
    flow = _build_flow()
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    session["oauth_state"] = state
    # Store PKCE code_verifier so /callback can use it
    session["code_verifier"] = flow.code_verifier
    return redirect(auth_url)


@app.route("/callback")
def callback():
    # Validate OAuth state to prevent CSRF
    if request.args.get("state") != session.get("oauth_state"):
        return "Invalid OAuth state. <a href='/login'>Try again</a>", 403

    flow = _build_flow()
    # Restore the PKCE code_verifier from the session
    flow.code_verifier = session.pop("code_verifier", None)
    flow.fetch_token(authorization_response=request.url)

    credentials = flow.credentials

    # Fetch user profile
    userinfo = http_requests.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {credentials.token}"},
        timeout=10,
    ).json()

    google_id = userinfo["id"]
    email = userinfo.get("email", "")
    name = userinfo.get("name", email)

    # Store user + tokens
    user_id = upsert_user(
        google_id=google_id,
        email=email,
        name=name,
        access_token=credentials.token,
        refresh_token=credentials.refresh_token,
        token_expiry=credentials.expiry.isoformat() if credentials.expiry else None,
    )

    # Clean up OAuth temp data
    session.pop("oauth_state", None)

    session["user_id"] = user_id
    session["user_email"] = email
    session["user_name"] = name

    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── Dashboard ──────────────────────────────────────────────────────────
@app.route("/")
@login_required
def dashboard():
    email_count = get_user_email_count(session["user_id"])
    return render_template_string(DASHBOARD_HTML,
        user_name=session.get("user_name", ""),
        user_email=session.get("user_email", ""),
        email_count=email_count,
    )


# ── API Endpoints ─────────────────────────────────────────────────────
@app.route("/fetch", methods=["POST"])
@login_required
def fetch_emails():
    _check_origin()
    try:
        user_id = session["user_id"]
        data = request.get_json(silent=True) or {}
        limit = data.get("limit", 100)

        # Validate and cap
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            limit = 100
        limit = max(1, min(limit, 500))

        service = authenticate_gmail(user_id)
        result = backfill(service, user_id, limit=limit)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/sync", methods=["POST"])
@login_required
def sync_emails():
    _check_origin()
    try:
        user_id = session["user_id"]
        service = authenticate_gmail(user_id)
        result = delta_sync(service, user_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/build", methods=["POST"])
@login_required
def build_index():
    _check_origin()
    try:
        user_id = session["user_id"]
        invalidate_user_cache(user_id)
        result = build_user_index(user_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    _check_origin()
    try:
        user_id = session["user_id"]
        data = request.get_json()
        question = data.get("question", "").strip()
        chat_history = data.get("history", [])
        if not question:
            return jsonify({"status": "error", "message": "Empty question"}), 400
        answer = ask_my_emails(user_id, question, chat_history=chat_history)
        return jsonify({"status": "ok", "answer": answer})
    except FileNotFoundError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/stats", methods=["GET"])
@login_required
def stats():
    """Return email count for the logged-in user."""
    count = get_user_email_count(session["user_id"])
    return jsonify({"email_count": count})


# ═══════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ═══════════════════════════════════════════════════════════════════════
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MailSense — Gmail RAG Assistant</title>
<meta name="description" content="Search and explore your Gmail inbox using AI-powered semantic retrieval.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

  :root{
    --bg:#0b0f1a;
    --surface:#111827;
    --surface2:#1a2235;
    --surface3:#212b42;
    --border:#253049;
    --text:#e2e8f0;
    --text-dim:#94a3b8;
    --accent:#6366f1;
    --accent-glow:rgba(99,102,241,.35);
    --green:#22c55e;
    --amber:#f59e0b;
    --red:#ef4444;
    --radius:14px;
  }

  body{
    font-family:'Inter',system-ui,sans-serif;
    background:var(--bg);
    color:var(--text);
    min-height:100vh;
    display:flex;
    flex-direction:column;
    align-items:center;
  }

  /* ─── Top Bar ─────────────────────────── */
  .topbar{
    width:100%;
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:16px 32px;
    background:var(--surface);
    border-bottom:1px solid var(--border);
    position:sticky;top:0;z-index:50;
    backdrop-filter:blur(12px);
  }
  .topbar .brand{font-size:18px;font-weight:700;letter-spacing:-.3px;display:flex;align-items:center;gap:8px}
  .topbar .brand span{background:linear-gradient(135deg,var(--accent),#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .topbar .user-info{display:flex;align-items:center;gap:12px;font-size:13px;color:var(--text-dim)}
  .topbar .avatar{width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,var(--accent),#a78bfa);display:flex;align-items:center;justify-content:center;font-weight:600;font-size:14px;color:#fff}
  .topbar a.logout{color:var(--text-dim);text-decoration:none;font-size:12px;border:1px solid var(--border);padding:5px 12px;border-radius:8px;transition:.2s}
  .topbar a.logout:hover{border-color:var(--accent);color:var(--accent)}

  /* ─── Main Layout ─────────────────────── */
  .main{width:100%;max-width:900px;padding:28px 20px;flex:1;display:flex;flex-direction:column;gap:20px}

  /* ─── Stats Bar ───────────────────────── */
  .stats-bar{
    display:flex;gap:12px;
  }
  .stat{
    flex:1;
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:var(--radius);
    padding:16px 20px;
    display:flex;align-items:center;gap:12px;
  }
  .stat-icon{
    width:40px;height:40px;border-radius:10px;
    display:flex;align-items:center;justify-content:center;
    font-size:18px;
  }
  .stat-icon.emails{background:rgba(99,102,241,.15)}
  .stat-label{font-size:11px;color:var(--text-dim);text-transform:uppercase;letter-spacing:.5px}
  .stat-value{font-size:22px;font-weight:700;margin-top:2px}

  /* ─── Cards ───────────────────────────── */
  .card{
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:var(--radius);
    padding:24px;
  }
  .card h2{font-size:15px;font-weight:600;margin-bottom:16px;display:flex;align-items:center;gap:8px}

  /* ─── Action Buttons ──────────────────── */
  .actions{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end}
  .btn{
    padding:10px 22px;
    border-radius:10px;
    border:none;
    font-family:inherit;
    font-size:13px;
    font-weight:600;
    cursor:pointer;
    transition:all .2s;
    display:flex;align-items:center;gap:6px;
    color:#fff;
    position:relative;
    overflow:hidden;
    white-space:nowrap;
  }
  .btn::after{
    content:'';position:absolute;inset:0;
    background:linear-gradient(180deg,rgba(255,255,255,.08) 0%,transparent 100%);
    pointer-events:none;
  }
  .btn:disabled{opacity:.5;cursor:not-allowed}
  .btn-primary{background:var(--accent);box-shadow:0 2px 16px var(--accent-glow)}
  .btn-primary:hover:not(:disabled){background:#5558e6;transform:translateY(-1px);box-shadow:0 4px 24px var(--accent-glow)}
  .btn-secondary{background:var(--surface2);border:1px solid var(--border);color:var(--text)}
  .btn-secondary:hover:not(:disabled){border-color:var(--accent);color:var(--accent)}
  .btn-build{background:linear-gradient(135deg,#6366f1,#a78bfa)}
  .btn-build:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 4px 24px rgba(167,139,250,.3)}

  /* ─── Fetch Limit Input ───────────────── */
  .fetch-group{display:flex;align-items:center;gap:8px}
  .fetch-input{
    width:72px;
    padding:9px 12px;
    border-radius:10px;
    border:1px solid var(--border);
    background:var(--surface2);
    color:var(--text);
    font-family:inherit;
    font-size:13px;
    font-weight:600;
    text-align:center;
    outline:none;
    transition:border .2s;
  }
  .fetch-input:focus{border-color:var(--accent)}
  .fetch-label{font-size:11px;color:var(--text-dim)}

  /* ─── Status Toast ────────────────────── */
  .status{
    margin-top:14px;
    padding:12px 16px;
    border-radius:10px;
    font-size:13px;
    font-weight:500;
    display:none;
    animation:fadeIn .3s;
  }
  .status.ok{display:block;background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.25);color:var(--green)}
  .status.error{display:block;background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.25);color:var(--red)}
  .status.loading{display:flex;align-items:center;gap:8px;background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.25);color:var(--accent)}

  @keyframes fadeIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:translateY(0)}}

  .spinner{width:16px;height:16px;border:2px solid transparent;border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}

  /* ─── Chat ────────────────────────────── */
  .chat-area{flex:1;display:flex;flex-direction:column;min-height:380px}
  .messages{
    flex:1;
    overflow-y:auto;
    display:flex;
    flex-direction:column;
    gap:12px;
    padding:4px 0;
    max-height:420px;
    scrollbar-width:thin;
    scrollbar-color:var(--border) transparent;
  }
  .msg{
    padding:12px 16px;
    border-radius:12px;
    font-size:13.5px;
    line-height:1.65;
    max-width:85%;
    animation:fadeIn .25s;
    white-space:pre-wrap;
    word-wrap:break-word;
  }
  .msg.user{
    background:var(--accent);
    color:#fff;
    align-self:flex-end;
    border-bottom-right-radius:4px;
  }
  .msg.assistant{
    background:var(--surface2);
    border:1px solid var(--border);
    align-self:flex-start;
    border-bottom-left-radius:4px;
  }

  .chat-input-row{
    display:flex;
    gap:10px;
    margin-top:14px;
  }
  .chat-input-row input{
    flex:1;
    padding:12px 16px;
    border-radius:10px;
    border:1px solid var(--border);
    background:var(--surface2);
    color:var(--text);
    font-family:inherit;
    font-size:14px;
    outline:none;
    transition:border .2s;
  }
  .chat-input-row input:focus{border-color:var(--accent)}
  .chat-input-row input::placeholder{color:var(--text-dim)}

  .btn-send{
    padding:12px 20px;
    background:var(--accent);
    border:none;
    border-radius:10px;
    color:#fff;
    font-family:inherit;
    font-weight:600;
    font-size:13px;
    cursor:pointer;
    transition:.2s;
    display:flex;align-items:center;gap:6px;
  }
  .btn-send:hover:not(:disabled){background:#5558e6}
  .btn-send:disabled{opacity:.5;cursor:not-allowed}

  /* ─── Footer ──────────────────────────── */
  .footer{text-align:center;padding:16px;font-size:11px;color:var(--text-dim);opacity:.6}

  /* ─── Responsive ──────────────────────── */
  @media(max-width:600px){
    .topbar{padding:12px 16px}
    .main{padding:16px 12px}
    .actions{flex-direction:column}
    .stats-bar{flex-direction:column}
    .msg{max-width:95%}
  }
</style>
</head>
<body>

<!-- ═══ Top Bar ═══ -->
<div class="topbar">
  <div class="brand">📬 <span>MailSense</span></div>
  <div class="user-info">
    <div class="avatar">{{ user_name[0] | upper }}</div>
    <div>
      <div style="font-weight:500;color:var(--text)">{{ user_name }}</div>
      <div style="font-size:11px">{{ user_email }}</div>
    </div>
    <a href="/logout" class="logout">Sign out</a>
  </div>
</div>

<!-- ═══ Main ═══ -->
<div class="main">

  <!-- Stats Bar -->
  <div class="stats-bar">
    <div class="stat">
      <div class="stat-icon emails">📧</div>
      <div>
        <div class="stat-label">Emails in DB</div>
        <div class="stat-value" id="emailCount">{{ email_count }}</div>
      </div>
    </div>
  </div>

  <!-- Data Pipeline Card -->
  <div class="card">
    <h2>⚡ Data Pipeline</h2>
    <div class="actions">
      <div class="fetch-group">
        <button class="btn btn-primary" id="btnFetch" onclick="doFetch()">
          📥 Fetch Emails
        </button>
        <input type="number" class="fetch-input" id="fetchLimit" value="100" min="1" max="500" title="Number of emails to fetch">
        <span class="fetch-label">limit</span>
      </div>
      <button class="btn btn-secondary" id="btnSync" onclick="doAction('/sync')">
        🔄 Delta Sync
      </button>
      <button class="btn btn-build" id="btnBuild" onclick="doAction('/build')">
        🧠 Process &amp; Build Index
      </button>
    </div>
    <div class="status" id="actionStatus"></div>
  </div>

  <!-- Chat Card -->
  <div class="card chat-area">
    <h2>💬 Ask Your Emails</h2>
    <div class="messages" id="messages"></div>
    <div class="chat-input-row">
      <input type="text" id="chatInput" placeholder="e.g. What did Google send me last week?" 
             onkeydown="if(event.key==='Enter')sendChat()">
      <button class="btn-send" id="btnSend" onclick="sendChat()">Send ➜</button>
    </div>
  </div>

</div>

<div class="footer">MailSense · Local Gmail RAG System · Your data never leaves your machine</div>

<script>
  // ─── Helpers ──────────────────────────────────
  const status = document.getElementById('actionStatus');
  const allBtns = () => document.querySelectorAll('.actions .btn');

  function setLoading(msg) {
    allBtns().forEach(b => b.disabled = true);
    status.className = 'status loading';
    status.innerHTML = '<div class="spinner"></div> ' + msg;
  }

  function setResult(data) {
    if (data.status === 'ok' || data.status === 'empty') {
      status.className = 'status ok';
      status.textContent = '✅ ' + data.message;
    } else {
      status.className = 'status error';
      status.textContent = '❌ ' + data.message;
    }
    allBtns().forEach(b => b.disabled = false);
    refreshStats();
  }

  function setError(err) {
    status.className = 'status error';
    status.textContent = '❌ ' + err;
    allBtns().forEach(b => b.disabled = false);
  }

  async function refreshStats() {
    try {
      const res = await fetch('/stats');
      const data = await res.json();
      document.getElementById('emailCount').textContent = data.email_count;
    } catch(e) {}
  }

  // ─── Fetch with custom limit ──────────────────
  async function doFetch() {
    const limit = parseInt(document.getElementById('fetchLimit').value) || 100;
    setLoading('Fetching ' + limit + ' emails…');
    try {
      const res = await fetch('/fetch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit: limit }),
      });
      setResult(await res.json());
    } catch (err) {
      setError('Network error: ' + err.message);
    }
  }

  // ─── Generic action (sync / build) ────────────
  async function doAction(endpoint) {
    const labels = { '/sync': 'Syncing new emails…', '/build': 'Building index… this may take a minute' };
    setLoading(labels[endpoint] || 'Working…');
    try {
      const res = await fetch(endpoint, { method: 'POST' });
      setResult(await res.json());
    } catch (err) {
      setError('Network error: ' + err.message);
    }
  }

  // ─── Chat ─────────────────────────────────────
  const messagesDiv = document.getElementById('messages');
  const chatInput = document.getElementById('chatInput');
  let chatHistory = [];  // tracks conversation for follow-ups

  function addMessage(role, text) {
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    div.textContent = text;
    messagesDiv.appendChild(div);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }

  async function sendChat() {
    const q = chatInput.value.trim();
    if (!q) return;

    addMessage('user', q);
    chatHistory.push({ role: 'user', content: q });
    chatInput.value = '';

    const btn = document.getElementById('btnSend');
    btn.disabled = true;
    chatInput.disabled = true;

    // Thinking indicator
    const thinking = document.createElement('div');
    thinking.className = 'msg assistant';
    thinking.innerHTML = '<div class="spinner" style="display:inline-block;vertical-align:middle;margin-right:8px"></div> Thinking…';
    messagesDiv.appendChild(thinking);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, history: chatHistory }),
      });
      const data = await res.json();

      messagesDiv.removeChild(thinking);

      if (data.status === 'ok') {
        addMessage('assistant', data.answer);
        chatHistory.push({ role: 'assistant', content: data.answer });
      } else {
        addMessage('assistant', '❌ ' + data.message);
      }
    } catch (err) {
      messagesDiv.removeChild(thinking);
      addMessage('assistant', '❌ Network error: ' + err.message);
    } finally {
      btn.disabled = false;
      chatInput.disabled = false;
      chatInput.focus();
    }
  }
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=5000)