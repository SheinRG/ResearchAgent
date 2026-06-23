"use client";

import { useState, useEffect, createContext, useContext, useCallback } from "react";
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

  // Load from localStorage on mount
  useEffect(() => {
    const storedToken = localStorage.getItem("auth_token");
    const storedUser = localStorage.getItem("auth_user");
    
    if (storedToken && storedUser) {
      setToken(storedToken);
      try {
        setUser(JSON.parse(storedUser));
      } catch (e) {
        console.error("Failed to parse user from local storage", e);
      }
    }
    setIsLoading(false);
  }, []);

  const saveAuth = (newToken, newUser) => {
    setToken(newToken);
    setUser(newUser);
    localStorage.setItem("auth_token", newToken);
    localStorage.setItem("auth_user", JSON.stringify(newUser));
  };

  const clearAuth = useCallback(() => {
    setToken(null);
    setUser(null);
    localStorage.removeItem("auth_token");
    localStorage.removeItem("auth_user");
    router.push("/login");
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
      return data.token;
    } catch {
      return null;
    }
  }, []);

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
