"use client";

import { usePathname, useRouter } from "next/navigation";
import Link from "next/link";
import { useState, useEffect, useCallback } from "react";
import { useTheme } from "next-themes";
import { useAuth } from "@/hooks/useAuth";
import { useAccent } from "@/components/AccentProvider";
import useResearchStore from "@/stores/researchStore";
import useToast from "@/stores/toastStore";

import {
  LogoMark,
  PlusIcon,
  MenuIcon,
  CloseIcon,
  LogoutIcon,
  PanelLeftIcon,
  NoteIcon,
  FileTextIcon,
  ClockIcon,
  ChevronRightIcon,
  SunIcon,
  MoonIcon,
  MonitorIcon,
  SwatchIcon,
  CheckIcon,
  TrashIcon,
  UserIcon,
} from "@/components/Icons";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function formatAgo(timestamp) {
  const minutes = Math.floor((Date.now() - timestamp) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return new Date(timestamp).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

const ACCENT_LABELS = { blue: "Blue", terracotta: "Terracotta", green: "Green" };

export default function AppLayout({ children }) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, isAuthenticated, token, logout, updateProfile } = useAuth();
  const { sessionsNonce } = useResearchStore();
  const { theme, resolvedTheme, setTheme } = useTheme();
  const { accent, setAccent } = useAccent();
  const showToast = useToast((s) => s.show);

  const [mounted, setMounted] = useState(false);
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [appearanceOpen, setAppearanceOpen] = useState(false);

  // Personalization settings modal.
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [nameDraft, setNameDraft] = useState("");
  const [savingName, setSavingName] = useState(false);

  // DB-backed session history for the sidebar.
  const [dbSessions, setDbSessions] = useState([]);

  // DB-backed notes for the sidebar.
  const [apiNotes, setApiNotes] = useState([]);

  // Hourly query usage.
  const [rateLimit, setRateLimit] = useState(null); // { used, limit, remaining }

  // Note modal: { id } where id=null means a new note.
  const [noteModal, setNoteModal] = useState(null);
  const [noteDraft, setNoteDraft] = useState("");

  useEffect(() => {
    setMounted(true);
    setIsCollapsed(localStorage.getItem("sidebar_collapsed") === "true");
  }, []);

  /**
   * Fetch the thread list from GET /api/sessions whenever:
   *   - the user logs in (token changes)
   *   - a research turn finishes (sessionsNonce bumps)
   *   - the user navigates back to "/" (pathname changes to "/")
   *
   * Failures are silently swallowed so the sidebar degrades gracefully.
   */
  const fetchSessions = useCallback(async () => {
    if (!token) return;
    try {
      const res = await fetch(`${API_BASE}/api/sessions?limit=20`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) return;
      const data = await res.json();
      setDbSessions(Array.isArray(data) ? data : []);
    } catch {
      // Network error — keep the last-known list visible.
    }
  }, [token]);

  /** Fetch notes from GET /api/notes. */
  const fetchNotes = useCallback(async () => {
    if (!token) return;
    try {
      const res = await fetch(`${API_BASE}/api/notes`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) return;
      const data = await res.json();
      setApiNotes(Array.isArray(data) ? data : []);
    } catch {
      // Network error — keep existing notes visible.
    }
  }, [token]);

  const fetchRateLimit = useCallback(async () => {
    if (!token) return;
    try {
      const res = await fetch(`${API_BASE}/api/auth/rate-limit`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) return;
      setRateLimit(await res.json());
    } catch {
      // silently ignore
    }
  }, [token]);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions, sessionsNonce, pathname]);

  useEffect(() => {
    fetchNotes();
  }, [fetchNotes]);

  // Refresh rate limit on login and after each research turn completes.
  useEffect(() => {
    fetchRateLimit();
  }, [fetchRateLimit, sessionsNonce]);

  const toggleCollapse = () =>
    setIsCollapsed((v) => {
      const next = !v;
      localStorage.setItem("sidebar_collapsed", String(next));
      return next;
    });

  const isLoginPage = pathname === "/login";
  const showSidebar = isAuthenticated && !isLoginPage;

  const handleNewThread = () => {
    router.push("/");
    setIsMobileOpen(false);
  };

  /** Navigate to a stored session thread (loads stored results, no re-run). */
  const openSession = (sessionId) => {
    router.push(`/research?session=${encodeURIComponent(sessionId)}`);
    setIsMobileOpen(false);
  };

  /** Delete a history thread (with confirmation), then drop it from the list. */
  const deleteSession = async (sessionId, e) => {
    e.stopPropagation();
    if (!window.confirm("Delete this thread? This can't be undone.")) return;
    try {
      const res = await fetch(
        `${API_BASE}/api/sessions/${encodeURIComponent(sessionId)}`,
        { method: "DELETE", headers: { Authorization: `Bearer ${token}` } }
      );
      if (!res.ok) throw new Error();
      setDbSessions((prev) => prev.filter((s) => s.id !== sessionId));
      showToast("Thread deleted");
      // If we're viewing the thread we just deleted, return home.
      if (pathname.startsWith("/research")) router.push("/");
    } catch {
      showToast("Failed to delete thread");
    }
  };

  /** Open the personalization settings modal, prefilled with the saved name. */
  const openSettings = () => {
    setNameDraft(user?.preferred_name || "");
    setSettingsOpen(true);
    closeProfile();
  };

  const saveSettings = async () => {
    setSavingName(true);
    const result = await updateProfile(nameDraft.trim());
    setSavingName(false);
    showToast(result.success ? "Personalization saved" : "Failed to save");
    if (result.success) setSettingsOpen(false);
  };

  const closeProfile = () => {
    setProfileOpen(false);
    setAppearanceOpen(false);
  };

  // --- Notes (API-backed) ---
  const openNewNote = () => {
    setNoteModal({ id: null });
    setNoteDraft("");
    setProfileOpen(false);
  };
  const openExistingNote = (note) => {
    setNoteModal({ id: note.id });
    setNoteDraft(note.text);
  };
  const saveNote = async () => {
    const text = noteDraft.trim();
    if (!text) {
      setNoteModal(null);
      return;
    }
    try {
      if (noteModal?.id) {
        // Update existing note
        const res = await fetch(`${API_BASE}/api/notes/${noteModal.id}`, {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ text }),
        });
        if (res.ok) {
          const updated = await res.json();
          setApiNotes((prev) =>
            prev.map((n) => (n.id === noteModal.id ? updated : n))
          );
        }
      } else {
        // Create new note
        const res = await fetch(`${API_BASE}/api/notes`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ text }),
        });
        if (res.ok) {
          const created = await res.json();
          setApiNotes((prev) => [created, ...prev]);
        }
      }
      showToast("Note saved");
    } catch {
      showToast("Failed to save note");
    }
    setNoteModal(null);
  };
  const removeNote = async () => {
    if (!noteModal?.id) {
      setNoteModal(null);
      return;
    }
    try {
      await fetch(`${API_BASE}/api/notes/${noteModal.id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      setApiNotes((prev) => prev.filter((n) => n.id !== noteModal.id));
      showToast("Note deleted");
    } catch {
      showToast("Failed to delete note");
    }
    setNoteModal(null);
  };

  // --- Appearance ---
  const appearanceLabel = !mounted
    ? ""
    : theme === "system"
    ? `System (${resolvedTheme === "dark" ? "Dark" : "Light"})`
    : theme === "dark"
    ? "Dark"
    : "Light";

  const AppearanceModeIcon =
    theme === "system" ? MonitorIcon : resolvedTheme === "dark" ? MoonIcon : SunIcon;

  if (!showSidebar) {
    return <div className="app-main-wrapper">{children}</div>;
  }

  const userInitial = (user?.name || user?.email || "?").charAt(0).toUpperCase();

  return (
    <div className={`layout-container ${isCollapsed ? "sidebar-collapsed" : ""}`}>
      {/* Desktop reopen button — shown only when collapsed */}
      {isCollapsed && (
        <button
          className="sidebar-reopen-btn"
          onClick={toggleCollapse}
          aria-label="Open goon.ai"
          title="Open goon.ai"
        >
          <LogoMark size={26} />
        </button>
      )}

      {/* Mobile header */}
      <header className="mobile-header">
        <button
          className="mobile-menu-toggle"
          onClick={() => setIsMobileOpen((v) => !v)}
          aria-label="Toggle navigation menu"
        >
          <MenuIcon />
        </button>
        <Link href="/" className="navbar-brand">
          <LogoMark size={22} />
          <span className="brand-text">
            goon<span className="wordmark-accent">.ai</span>
          </span>
        </Link>
      </header>

      {/* Sidebar */}
      <div className={`sidebar-shell ${isMobileOpen ? "mobile-open" : ""}`}>
        <aside className="sidebar-container">
          <div className="sidebar-brand">
            <Link href="/" className="navbar-brand">
              <LogoMark size={22} />
              <span className="brand-text">
                goon<span className="wordmark-accent">.ai</span>
              </span>
            </Link>
            <button
              className="sidebar-icon-btn"
              onClick={toggleCollapse}
              title="Collapse"
              aria-label="Collapse sidebar"
            >
              <PanelLeftIcon width={17} height={17} />
            </button>
            <button
              className="mobile-close-btn"
              onClick={() => setIsMobileOpen(false)}
              aria-label="Close navigation menu"
            >
              <CloseIcon width={18} height={18} />
            </button>
          </div>

          <button className="sidebar-new-btn" onClick={handleNewThread}>
            <PlusIcon width={16} height={16} />
            New thread
          </button>

          <button className="sidebar-ghost-btn" onClick={openNewNote}>
            <NoteIcon width={15} height={15} />
            Add note
          </button>

          <div className="sidebar-nav">
            {apiNotes.length > 0 && (
              <>
                <div className="sidebar-section-label">
                  <FileTextIcon width={12} height={12} />
                  Notes
                </div>
                {apiNotes.map((note) => (
                  <button
                    key={note.id}
                    className="sidebar-list-item"
                    onClick={() => openExistingNote(note)}
                  >
                    <span className="sidebar-list-title">
                      {note.text.split("\n")[0].slice(0, 42) || "Untitled note"}
                    </span>
                    <span className="sidebar-list-time">
                      {formatAgo(new Date(note.updated_at).getTime())}
                    </span>
                  </button>
                ))}
                <div className="sidebar-section-divider" />
              </>
            )}

            <div className="sidebar-section-label">
              <ClockIcon width={12} height={12} />
              History
            </div>
            {dbSessions.length === 0 ? (
              <div className="sidebar-empty">
                No history yet. Ask something to get started.
              </div>
            ) : (
              dbSessions.map((session) => (
                <div key={session.id} className="sidebar-list-row">
                  <button
                    className="sidebar-list-item"
                    onClick={() => openSession(session.id)}
                  >
                    <span className="sidebar-list-title">{session.title || session.query}</span>
                    <span className="sidebar-list-time">
                      {/* updated_at is an ISO string; convert to ms for formatAgo */}
                      {session.updated_at
                        ? formatAgo(new Date(session.updated_at).getTime())
                        : formatAgo(new Date(session.created_at).getTime())}
                    </span>
                  </button>
                  <button
                    className="sidebar-item-delete"
                    onClick={(e) => deleteSession(session.id, e)}
                    title="Delete thread"
                    aria-label="Delete thread"
                  >
                    <TrashIcon width={14} height={14} />
                  </button>
                </div>
              ))
            )}
          </div>

          {/* Profile footer */}
          <div className="sidebar-footer">
            {rateLimit && (() => {
              const usedRatio = rateLimit.limit ? rateLimit.used / rateLimit.limit : 0;
              const fillTone =
                usedRatio >= 0.9 ? " rate-limit-fill--danger"
                : usedRatio >= 0.7 ? " rate-limit-fill--warn"
                : "";
              return (
                <div className="rate-limit-bar">
                  <div className="rate-limit-track">
                    <div
                      className={`rate-limit-fill${fillTone}`}
                      style={{ width: `${Math.min(100, usedRatio * 100)}%` }}
                    />
                  </div>
                  <span className="rate-limit-label">
                    {rateLimit.remaining} / {rateLimit.limit} queries left this hour
                  </span>
                </div>
              );
            })()}
            <button
              className="profile-trigger"
              onClick={() => setProfileOpen((v) => !v)}
            >
              <span className="user-avatar">{userInitial}</span>
              <span className="user-info">
                <span className="user-name">{user?.name || "Researcher"}</span>
                <span className="user-email">{user?.email}</span>
              </span>
              <ChevronRightIcon
                width={15}
                height={15}
                style={{ transform: "rotate(-90deg)", color: "var(--text-tertiary)" }}
              />
            </button>

            {profileOpen && (
              <>
                <div className="menu-backdrop" onClick={closeProfile} />
                <div className="popup-menu profile-menu">
                  <div className="profile-menu-head">
                    <span className="user-avatar">{userInitial}</span>
                    <span className="user-info">
                      <span className="user-name">{user?.name || "Researcher"}</span>
                      <span className="user-email">{user?.email}</span>
                    </span>
                  </div>
                  <div className="menu-divider" />

                  <button className="menu-item" onClick={openSettings}>
                    <UserIcon width={16} height={16} />
                    <span className="menu-item-grow">
                      Personalization
                      <span className="menu-item-sub">
                        {user?.preferred_name
                          ? `Called "${user.preferred_name}"`
                          : "Set what goon calls you"}
                      </span>
                    </span>
                    <ChevronRightIcon
                      width={14}
                      height={14}
                      className="menu-item-chevron"
                    />
                  </button>

                  <div className="menu-item-with-flyout">
                    <button
                      className="menu-item"
                      onClick={() => setAppearanceOpen((v) => !v)}
                    >
                      <AppearanceModeIcon width={16} height={16} />
                      <span className="menu-item-grow">
                        Appearance
                        <span className="menu-item-sub">{appearanceLabel}</span>
                      </span>
                      <ChevronRightIcon
                        width={14}
                        height={14}
                        className="menu-item-chevron"
                        style={{
                          transform: appearanceOpen ? "rotate(180deg)" : "none",
                        }}
                      />
                    </button>

                    {appearanceOpen && (
                      <div className="popup-menu appearance-flyout">
                        {[
                          { id: "light", label: "Light", Icon: SunIcon },
                          { id: "dark", label: "Dark", Icon: MoonIcon },
                          { id: "system", label: "System", Icon: MonitorIcon },
                        ].map(({ id, label, Icon }) => (
                          <button
                            key={id}
                            className="menu-item"
                            onClick={() => setTheme(id)}
                          >
                            <Icon width={15} height={15} />
                            <span className="menu-item-grow">{label}</span>
                            {theme === id && (
                              <CheckIcon
                                width={15}
                                height={15}
                                className="menu-item-check"
                              />
                            )}
                          </button>
                        ))}

                        <div className="menu-divider" />

                        <div className="accent-row">
                          <SwatchIcon
                            width={15}
                            height={15}
                            style={{ color: "var(--text-secondary)", marginRight: 4 }}
                          />
                          {["blue", "terracotta", "green"].map((a) => (
                            <button
                              key={a}
                              className={`accent-swatch accent-${a} ${
                                accent === a ? "is-active" : ""
                              }`}
                              onClick={() => setAccent(a)}
                              title={ACCENT_LABELS[a]}
                              aria-label={`${ACCENT_LABELS[a]} accent`}
                            />
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="menu-divider" />
                  <button
                    className="menu-item"
                    onClick={() => {
                      closeProfile();
                      logout();
                    }}
                  >
                    <LogoutIcon width={16} height={16} />
                    <span className="menu-item-grow">Sign out</span>
                  </button>
                </div>
              </>
            )}
          </div>
        </aside>
      </div>

      {/* Mobile backdrop */}
      {isMobileOpen && (
        <div
          className="mobile-sidebar-backdrop"
          onClick={() => setIsMobileOpen(false)}
        />
      )}

      {/* Main content */}
      <main className="layout-content-viewport">{children}</main>

      {/* Note modal */}
      {noteModal && (
        <div className="modal-backdrop" onClick={() => setNoteModal(null)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <span className="modal-title">
                {noteModal.id ? "Edit note" : "New note"}
              </span>
              <button
                className="msg-action-btn"
                onClick={() => setNoteModal(null)}
                aria-label="Close"
              >
                <CloseIcon width={18} height={18} />
              </button>
            </div>
            <textarea
              className="modal-textarea"
              rows={6}
              value={noteDraft}
              onChange={(e) => setNoteDraft(e.target.value)}
              placeholder="Jot down a thought, a finding, a to-do…"
              autoFocus
            />
            <div className="modal-foot">
              {noteModal.id && (
                <button className="btn-danger-text" onClick={removeNote}>
                  Delete
                </button>
              )}
              <div className="modal-actions">
                <button className="btn-ghost" onClick={() => setNoteModal(null)}>
                  Cancel
                </button>
                <button className="btn-accent" onClick={saveNote}>
                  Save note
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Personalization settings modal */}
      {settingsOpen && (
        <div className="modal-backdrop" onClick={() => setSettingsOpen(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <span className="modal-title">Personalization</span>
              <button
                className="msg-action-btn"
                onClick={() => setSettingsOpen(false)}
                aria-label="Close"
              >
                <CloseIcon width={18} height={18} />
              </button>
            </div>
            <label className="settings-label" htmlFor="preferred-name-input">
              What should goon call you?
            </label>
            <input
              id="preferred-name-input"
              className="modal-input"
              type="text"
              maxLength={50}
              value={nameDraft}
              onChange={(e) => setNameDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !savingName) saveSettings();
              }}
              placeholder="e.g. Rashi"
              autoFocus
            />
            <p className="settings-hint">
              The agent will address you by this name in its answers. Leave blank
              to turn it off.
            </p>
            <div className="modal-foot">
              <div className="modal-actions">
                <button
                  className="btn-ghost"
                  onClick={() => setSettingsOpen(false)}
                >
                  Cancel
                </button>
                <button
                  className="btn-accent"
                  onClick={saveSettings}
                  disabled={savingName}
                >
                  {savingName ? "Saving…" : "Save"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
