const state = {
  user: null,
  csrfToken: null,
  orgs: [],
  currentOrgId: null,
  threads: [],
  currentThreadId: null,
  pendingInviteToken: null,
  pollTimer: null,
};

const el = {
  toast: document.getElementById("toast"),
  authScreen: document.getElementById("auth-screen"),
  workspace: document.getElementById("workspace"),
  loginForm: document.getElementById("login-form"),
  registerForm: document.getElementById("register-form"),
  logoutBtn: document.getElementById("logout-btn"),
  userAvatar: document.getElementById("user-avatar"),
  userName: document.getElementById("user-name"),
  userEmail: document.getElementById("user-email"),
  orgList: document.getElementById("org-list"),
  newOrgForm: document.getElementById("new-org-form"),
  threadList: document.getElementById("thread-list"),
  newChatBtn: document.getElementById("new-chat-btn"),
  chatTitle: document.getElementById("chat-title"),
  chatSubtitle: document.getElementById("chat-subtitle"),
  messageList: document.getElementById("message-list"),
  composerForm: document.getElementById("composer-form"),
  composerInput: document.getElementById("composer-input"),
  correctionToggle: document.getElementById("correction-toggle"),
  modelVersion: document.getElementById("model-version"),
  modelStatus: document.getElementById("model-status"),
  modelPath: document.getElementById("model-path"),
  jobList: document.getElementById("job-list"),
  memberList: document.getElementById("member-list"),
  inviteForm: document.getElementById("invite-form"),
  inviteResult: document.getElementById("invite-result"),
};


function showToast(message, timeout = 2600) {
  el.toast.textContent = message;
  el.toast.classList.remove("hidden");
  window.clearTimeout(showToast._timer);
  showToast._timer = window.setTimeout(() => {
    el.toast.classList.add("hidden");
  }, timeout);
}


async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  const method = (options.method || "GET").toUpperCase();
  if (!headers["content-type"] && options.body !== undefined) {
    headers["content-type"] = "application/json";
  }
  if (!["GET", "HEAD", "OPTIONS"].includes(method) && state.csrfToken) {
    headers["x-csrf-token"] = state.csrfToken;
  }

  const response = await fetch(path, {
    ...options,
    method,
    credentials: "include",
    headers,
    body:
      options.body !== undefined && typeof options.body !== "string"
        ? JSON.stringify(options.body)
        : options.body,
  });

  if (response.status === 204) {
    return null;
  }

  let data = null;
  try {
    data = await response.json();
  } catch (_err) {
    data = null;
  }

  if (!response.ok) {
    throw new Error(data?.detail || `Request failed (${response.status})`);
  }
  return data;
}


function setAuthedUI(authed) {
  el.authScreen.classList.toggle("hidden", authed);
  el.workspace.classList.toggle("hidden", !authed);
}


function bindEvents() {
  el.loginForm.addEventListener("submit", onLogin);
  el.registerForm.addEventListener("submit", onRegister);
  el.logoutBtn.addEventListener("click", onLogout);
  el.newOrgForm.addEventListener("submit", onCreateOrg);
  el.newChatBtn.addEventListener("click", onCreateThread);
  el.composerForm.addEventListener("submit", onSendMessage);
  el.inviteForm.addEventListener("submit", onCreateInvite);
}


async function onLogin(event) {
  event.preventDefault();
  const formData = new FormData(event.currentTarget);
  try {
    const data = await api("/api/auth/login", {
      method: "POST",
      body: {
        email: formData.get("email"),
        password: formData.get("password"),
      },
    });
    await onAuthSuccess(data, "Welcome back");
  } catch (error) {
    showToast(error.message);
  }
}


async function onRegister(event) {
  event.preventDefault();
  const formData = new FormData(event.currentTarget);
  try {
    const data = await api("/api/auth/register", {
      method: "POST",
      body: {
        full_name: formData.get("full_name"),
        email: formData.get("email"),
        password: formData.get("password"),
      },
    });
    await onAuthSuccess(data, "Account created");
  } catch (error) {
    showToast(error.message);
  }
}


async function onAuthSuccess(data, toastText) {
  state.user = data.user;
  state.csrfToken = data.csrf_token;
  hydrateIdentity();
  setAuthedUI(true);
  await loadOrganizations();
  showToast(toastText);
  if (state.pendingInviteToken) {
    await acceptInvite(state.pendingInviteToken);
  }
}


async function onLogout() {
  try {
    await api("/api/auth/logout", { method: "POST" });
  } catch (_error) {
  }
  state.user = null;
  state.csrfToken = null;
  state.orgs = [];
  state.threads = [];
  state.currentOrgId = null;
  state.currentThreadId = null;
  clearInterval(state.pollTimer);
  setAuthedUI(false);
  renderOrgList();
  renderThreadList();
  renderMessages([]);
  showToast("Logged out");
}


function hydrateIdentity() {
  if (!state.user) {
    return;
  }
  el.userName.textContent = state.user.full_name;
  el.userEmail.textContent = state.user.email;
  el.userAvatar.textContent = (state.user.full_name || "U").slice(0, 1).toUpperCase();
}


async function restoreSession() {
  try {
    const data = await api("/api/auth/me");
    state.user = data.user;
    state.csrfToken = data.csrf_token;
    hydrateIdentity();
    setAuthedUI(true);
    await loadOrganizations();
  } catch (_error) {
    setAuthedUI(false);
  }
}


function orgNameById(orgId) {
  const org = state.orgs.find((entry) => entry.id === orgId);
  return org ? org.name : "";
}


async function loadOrganizations() {
  const orgs = await api("/api/orgs");
  state.orgs = orgs;
  renderOrgList();

  if (!state.currentOrgId || !state.orgs.some((org) => org.id === state.currentOrgId)) {
    state.currentOrgId = state.orgs[0]?.id ?? null;
  }

  if (state.currentOrgId) {
    await refreshOrgContext();
  } else {
    state.threads = [];
    state.currentThreadId = null;
    renderThreadList();
    renderMessages([]);
    el.chatTitle.textContent = "No organization yet";
    el.chatSubtitle.textContent = "Create your first organization from the left panel.";
    clearInterval(state.pollTimer);
  }
}


function renderOrgList() {
  el.orgList.innerHTML = "";
  state.orgs.forEach((org) => {
    const li = document.createElement("li");
    li.className = `list-item ${org.id === state.currentOrgId ? "active" : ""}`;
    li.textContent = `${org.name} (${org.role})`;
    li.addEventListener("click", async () => {
      state.currentOrgId = org.id;
      state.currentThreadId = null;
      renderOrgList();
      await refreshOrgContext();
    });
    el.orgList.appendChild(li);
  });
}


async function onCreateOrg(event) {
  event.preventDefault();
  const formData = new FormData(event.currentTarget);
  const name = String(formData.get("name") || "").trim();
  if (!name) {
    return;
  }
  try {
    const org = await api("/api/orgs", {
      method: "POST",
      body: { name },
    });
    event.currentTarget.reset();
    state.currentOrgId = org.id;
    await loadOrganizations();
    showToast("Organization created");
  } catch (error) {
    showToast(error.message);
  }
}


async function refreshOrgContext() {
  if (!state.currentOrgId) {
    return;
  }
  await Promise.all([
    loadThreads(),
    loadModelSnapshot(),
    loadMembers(),
    loadJobs(),
  ]);

  clearInterval(state.pollTimer);
  state.pollTimer = window.setInterval(() => {
    if (state.currentOrgId) {
      loadJobs();
      loadModelSnapshot();
    }
  }, 9000);
}


async function loadThreads() {
  if (!state.currentOrgId) {
    return;
  }
  const threads = await api(`/api/orgs/${state.currentOrgId}/threads`);
  state.threads = threads;

  if (!state.currentThreadId || !threads.some((thread) => thread.id === state.currentThreadId)) {
    state.currentThreadId = threads[0]?.id ?? null;
  }

  renderThreadList();
  if (state.currentThreadId) {
    await loadMessages();
  } else {
    renderMessages([]);
    el.chatTitle.textContent = `${orgNameById(state.currentOrgId)} - no chat yet`;
    el.chatSubtitle.textContent = "Create a thread and start talking to your org model.";
  }
}


function renderThreadList() {
  el.threadList.innerHTML = "";
  state.threads.forEach((thread) => {
    const li = document.createElement("li");
    li.className = `list-item ${thread.id === state.currentThreadId ? "active" : ""}`;
    li.textContent = thread.title;
    li.addEventListener("click", async () => {
      state.currentThreadId = thread.id;
      renderThreadList();
      await loadMessages();
    });
    el.threadList.appendChild(li);
  });
}


async function onCreateThread() {
  if (!state.currentOrgId) {
    showToast("Create an organization first");
    return;
  }
  const title = `New Chat ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
  try {
    const thread = await api(`/api/orgs/${state.currentOrgId}/threads`, {
      method: "POST",
      body: { title },
    });
    state.currentThreadId = thread.id;
    await loadThreads();
  } catch (error) {
    showToast(error.message);
  }
}


async function loadMessages() {
  if (!state.currentThreadId) {
    renderMessages([]);
    return;
  }
  const messages = await api(`/api/threads/${state.currentThreadId}/messages`);
  renderMessages(messages);
  const activeThread = state.threads.find((thread) => thread.id === state.currentThreadId);
  el.chatTitle.textContent = activeThread ? activeThread.title : "Chat";
  el.chatSubtitle.textContent = `Organization: ${orgNameById(state.currentOrgId)}`;
}


function renderMessages(messages) {
  el.messageList.innerHTML = "";
  if (!messages.length) {
    const empty = document.createElement("div");
    empty.className = "message assistant";
    empty.textContent = "No messages yet. Ask a question to begin.";
    el.messageList.appendChild(empty);
    return;
  }

  messages.forEach((message) => {
    const node = document.createElement("article");
    node.className = `message ${message.role}`;
    node.textContent = message.content;

    const meta = document.createElement("div");
    meta.className = "meta";
    const timestamp = new Date(message.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    meta.textContent = `${message.role} · ${timestamp}${message.is_correction ? " · correction" : ""}`;
    node.appendChild(meta);

    el.messageList.appendChild(node);
  });

  el.messageList.scrollTop = el.messageList.scrollHeight;
}


async function onSendMessage(event) {
  event.preventDefault();
  const content = el.composerInput.value.trim();
  if (!content) {
    return;
  }

  if (!state.currentOrgId) {
    showToast("Create or select an organization first");
    return;
  }

  if (!state.currentThreadId) {
    await onCreateThread();
    if (!state.currentThreadId) {
      showToast("Could not create a thread");
      return;
    }
  }

  const correction = el.correctionToggle.checked;

  try {
    const result = await api(`/api/threads/${state.currentThreadId}/messages`, {
      method: "POST",
      body: {
        content,
        is_correction: correction,
        trigger_training: correction,
      },
    });
    el.composerInput.value = "";
    el.correctionToggle.checked = false;
    await loadMessages();
    await loadThreads();
    if (result.training_job) {
      showToast(`Training job #${result.training_job.id} queued`);
      loadJobs();
    }
  } catch (error) {
    showToast(error.message);
  }
}


async function loadModelSnapshot() {
  if (!state.currentOrgId) {
    return;
  }
  try {
    const snapshot = await api(`/api/orgs/${state.currentOrgId}/model`);
    const active = snapshot.active_model_version;
    if (!active) {
      el.modelVersion.textContent = "-";
      el.modelStatus.innerHTML = "-";
      el.modelPath.textContent = "-";
      return;
    }
    el.modelVersion.textContent = `v${active.version_number}`;
    el.modelStatus.innerHTML = `<span class="status-pill ${active.status}">${active.status}</span>`;
    el.modelPath.textContent = active.model_path;
  } catch (error) {
    showToast(error.message);
  }
}


async function loadJobs() {
  if (!state.currentOrgId) {
    return;
  }
  try {
    const jobs = await api(`/api/orgs/${state.currentOrgId}/jobs`);
    el.jobList.innerHTML = "";
    if (!jobs.length) {
      const li = document.createElement("li");
      li.className = "list-item";
      li.textContent = "No training jobs yet.";
      el.jobList.appendChild(li);
      return;
    }
    jobs.forEach((job) => {
      const li = document.createElement("li");
      li.className = "list-item";
      const createdAt = new Date(job.created_at).toLocaleString([], {
        month: "short",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      });
      li.innerHTML = `
        <div>#${job.id} · ${createdAt}</div>
        <div><span class="status-pill ${job.status}">${job.status}</span></div>
        ${job.error_message ? `<small>${job.error_message.slice(0, 120)}</small>` : ""}
      `;
      el.jobList.appendChild(li);
    });
  } catch (error) {
    showToast(error.message);
  }
}


async function loadMembers() {
  if (!state.currentOrgId) {
    return;
  }
  try {
    const members = await api(`/api/orgs/${state.currentOrgId}/members`);
    el.memberList.innerHTML = "";
    members.forEach((member) => {
      const li = document.createElement("li");
      li.className = "list-item";
      li.innerHTML = `<strong>${member.full_name}</strong><br><small>${member.email} · ${member.role}</small>`;
      el.memberList.appendChild(li);
    });
  } catch (error) {
    showToast(error.message);
  }
}


async function onCreateInvite(event) {
  event.preventDefault();
  if (!state.currentOrgId) {
    showToast("Select an organization first");
    return;
  }
  const formData = new FormData(event.currentTarget);
  try {
    const invite = await api(`/api/orgs/${state.currentOrgId}/invites`, {
      method: "POST",
      body: {
        email: formData.get("email"),
        role: formData.get("role"),
      },
    });
    el.inviteResult.textContent = invite.invite_url;
    showToast("Invite created. Share the generated link.");
    event.currentTarget.reset();
  } catch (error) {
    showToast(error.message);
  }
}


async function acceptInvite(token) {
  try {
    const org = await api("/api/invites/accept", {
      method: "POST",
      body: { token },
    });
    state.pendingInviteToken = null;
    state.currentOrgId = org.id;
    await loadOrganizations();
    showToast(`Joined ${org.name}`);
    if (window.location.pathname.startsWith("/invite/")) {
      window.history.replaceState({}, "", "/");
    }
  } catch (error) {
    if (error.message.includes("Authentication required")) {
      showToast("Sign in first to accept your invite");
      return;
    }
    showToast(error.message);
  }
}


function detectInviteToken() {
  const fromData = document.body.dataset.inviteToken;
  if (fromData) {
    return fromData;
  }
  const pathParts = window.location.pathname.split("/").filter(Boolean);
  if (pathParts.length === 2 && pathParts[0] === "invite") {
    return pathParts[1];
  }
  return null;
}


async function boot() {
  bindEvents();
  state.pendingInviteToken = detectInviteToken();
  await restoreSession();
  if (state.user && state.pendingInviteToken) {
    await acceptInvite(state.pendingInviteToken);
  }
}


boot();
