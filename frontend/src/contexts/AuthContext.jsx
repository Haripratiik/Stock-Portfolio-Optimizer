import { createContext, useContext, useState, useEffect } from 'react'
import { auth, googleProvider } from '../firebase'
import {
  signInWithPopup,
  signOut,
  onAuthStateChanged,
} from 'firebase/auth'

const AuthContext = createContext(null)

const ALLOWED_EMAIL = import.meta.env.VITE_ALLOWED_EMAIL || ''

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [signingIn, setSigningIn] = useState(false)

  useEffect(() => {
    const unsub = onAuthStateChanged(auth, (firebaseUser) => {
      if (firebaseUser) {
        // Enforce email whitelist
        if (ALLOWED_EMAIL && firebaseUser.email !== ALLOWED_EMAIL) {
          signOut(auth)
          setUser(null)
          setError('Access denied. This account is not authorised.')
        } else {
          setUser(firebaseUser)
          setError('')
        }
      } else {
        setUser(null)
      }
      setLoading(false)
    })
    return unsub
  }, [])

  const login = async () => {
    if (signingIn) return
    try {
      setError('')
      setSigningIn(true)
      const result = await signInWithPopup(auth, googleProvider)
      if (ALLOWED_EMAIL && result.user.email !== ALLOWED_EMAIL) {
        await signOut(auth)
        setError('Access denied. This account is not authorised.')
      }
    } catch (err) {
      if (err.code === 'auth/cancelled-popup-request') {
        // User closed popup or clicked sign-in again — ignore
        setError('')
      } else if (err.code === 'auth/popup-blocked') {
        setError('Popup blocked. Allow popups for this site and try again.')
      } else {
        setError(err.message || 'Sign-in failed. Try again.')
      }
    } finally {
      setSigningIn(false)
    }
  }

  const logout = () => signOut(auth)

  return (
    <AuthContext.Provider value={{ user, loading, error, login, logout, signingIn }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
