"use client";

import { useState, useEffect, useRef, createContext, useContext, useCallback } from "react";
import { useRouter } from "next/navigation";

const AuthContext = createContext(null);
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Dev-only convenience: when the backend is unreachable on localhost, fall back
// to a mock session so the UI can be worked on offline. This MUST stay gated to
// development — in production a fake login would silently grant fake access.
const IS_DEV = process.env.NODE_ENV === "development";

const isNetworkError = (error) =>
  error.message?.includes("fetch") ||
  error.message?.includes("Failed") ||
  error.message?.includes("NetworkError");

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();
  // React 18 StrictMode mounts effects twice in dev; the bootstrap must run
  // once or the second pass could rotate/invalidate the session mid-flight.
  const bootstrappedRef = useRef(false);

  const saveAuth = (newToken, newUser) => {
    setToken(newToken);
    setUser(newUser);
    localStorage.setItem("auth_token", newToken);
    localStorage.setItem("auth_user", JSON.stringify(newUser));
    // One-shot flag for a *fresh* login (real login/register/Google sign-in).
    // The home greeting reads + clears it to play a one-time welcome. The silent
    // token refresh clears it below, and the on-mount localStorage restore never
    // calls saveAuth — so only a deliberate sign-in trips it.
    try {
      sessionStorage.setItem("just_logged_in", "1");
    } catch {
      // sessionStorage unavailable (private mode / SSR) — welcome is non-critical.
    }
  };

  const clearAuth = useCallback(({ redirect = true } = {}) => {
    setToken(null);
    setUser(null);
    localStorage.removeItem("auth_token");
    localStorage.removeItem("auth_user");
    // The session bootstrap clears without redirecting so public pages (shared
    // ?session= links) stay viewable; each guarded page routes to /login itself.
    if (redirect) router.push("/login");
  }, [router]);

  const login = async (email, password) => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email, password }),
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.detail || "Login failed");
      }
      
      saveAuth(data.token, data.user);
      return { success: true };
    } catch (error) {
      if (IS_DEV && isNetworkError(error)) {
        console.warn("[dev] Backend offline — using local mock session.");
        const mockUser = { id: "dev-user-id", email, name: email.split("@")[0] };
        saveAuth("dev-mock-jwt-token", mockUser);
        return { success: true };
      }
      if (isNetworkError(error)) {
        return { success: false, error: "Can't reach the server. Please try again shortly." };
      }
      return { success: false, error: error.message };
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (email, password, name) => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email, password, name }),
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.detail || "Registration failed");
      }
      
      saveAuth(data.token, data.user);
      return { success: true };
    } catch (error) {
      if (IS_DEV && isNetworkError(error)) {
        console.warn("[dev] Backend offline — using local mock session.");
        const mockUser = { id: "dev-user-id", email, name: name || email.split("@")[0] };
        saveAuth("dev-mock-jwt-token", mockUser);
        return { success: true };
      }
      if (isNetworkError(error)) {
        return { success: false, error: "Can't reach the server. Please try again shortly." };
      }
      return { success: false, error: error.message };
    } finally {
      setIsLoading(false);
    }
  };

  const loginWithGoogle = async (credential) => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/auth/google`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ credential }),
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.detail || "Google login failed");
      }
      
      saveAuth(data.token, data.user);
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    } finally {
      setIsLoading(false);
    }
  };

  const logout = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/api/auth/logout`, {
        method: "POST",
        credentials: "include",
      });
    } catch {
      // ignore — clear local state regardless
    }
    clearAuth();
  }, [clearAuth]);

  // Silently exchange the refresh token cookie for a new access token.
  // Returns the new token string on success, or null if the session has expired.
  const refreshSession = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/auth/refresh`, {
        method: "POST",
        credentials: "include",
      });
      if (!res.ok) return null;
      const data = await res.json();
      saveAuth(data.token, data.user);
      // A silent refresh is not a fresh login — don't replay the welcome.
      try {
        sessionStorage.removeItem("just_logged_in");
      } catch {
        // ignore
      }
      return data.token;
    } catch {
      return null;
    }
  }, []);

  // Session bootstrap: restore from localStorage on mount, but *validate* the
  // stored token against the backend before treating the user as signed in.
  // Previously the token was trusted blindly, so an expired token showed the
  // dashboard and then bounced the user to /login on their first query.
  useEffect(() => {
    if (bootstrappedRef.current) return;
    bootstrappedRef.current = true;

    const storedToken = localStorage.getItem("auth_token");
    const storedUser = localStorage.getItem("auth_user");
    if (!storedToken || !storedUser) {
      setIsLoading(false);
      return;
    }

    let parsedUser = null;
    try {
      parsedUser = JSON.parse(storedUser);
    } catch (e) {
      console.error("Failed to parse user from local storage", e);
    }

    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/auth/me`, {
          headers: { Authorization: `Bearer ${storedToken}` },
        });
        if (res.ok) {
          const fresh = await res.json();
          const merged = { ...(parsedUser || {}), ...fresh };
          setToken(storedToken);
          setUser(merged);
          localStorage.setItem("auth_user", JSON.stringify(merged));
        } else if (res.status === 401) {
          // Access token expired/invalid — silently exchange the refresh
          // cookie for a new one. Only a dead refresh token ends the session.
          const newToken = await refreshSession();
          if (!newToken) clearAuth({ redirect: false });
        } else {
          // Server hiccup (5xx etc.) — keep the session optimistically; the
          // per-request 401 handling will sort it out once the API recovers.
          setToken(storedToken);
          if (parsedUser) setUser(parsedUser);
        }
      } catch {
        // Backend unreachable (offline / dev without docker) — keep the stored
        // session so the UI remains usable; requests will retry when it's back.
        setToken(storedToken);
        if (parsedUser) setUser(parsedUser);
      } finally {
        setIsLoading(false);
      }
    })();
  }, [refreshSession, clearAuth]);

  // Update personalization (e.g. preferred name) and sync local state + storage.
  const updateProfile = useCallback(
    async (preferredName) => {
      try {
        const res = await fetch(`${API_BASE}/api/auth/profile`, {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ preferred_name: preferredName }),
        });
        if (!res.ok) throw new Error("Failed to update profile");
        const updated = await res.json();
        setUser((prev) => {
          const merged = { ...(prev || {}), ...updated };
          localStorage.setItem("auth_user", JSON.stringify(merged));
          return merged;
        });
        return { success: true };
      } catch (error) {
        return { success: false, error: error.message };
      }
    },
    [token]
  );

  const value = {
    user,
    token,
    isAuthenticated: !!token,
    isLoading,
    login,
    register,
    loginWithGoogle,
    logout,
    refreshSession,
    updateProfile,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
