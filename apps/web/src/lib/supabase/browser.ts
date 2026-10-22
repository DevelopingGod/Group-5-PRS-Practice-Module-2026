// Browser-side Supabase: anonymous sign-in only, so each device gets a pseudonymous user.
// The session lives in cookies, where the route handlers read it. No data is written from here.
import { createBrowserClient } from "@supabase/ssr";

let signedIn: Promise<void> | null = null;

function browserClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  return url && key ? createBrowserClient(url, key) : null;
}

/** Signs in anonymously once. Never throws: scanning works even if this fails. */
export function ensureSignedIn(): Promise<void> {
  signedIn ??= (async () => {
    const client = browserClient();
    if (!client) return;
    try {
      const { data } = await client.auth.getSession();
      if (data.session) return;
      const { error } = await client.auth.signInAnonymously();
      if (error) signedIn = null;
    } catch {
      signedIn = null; // try again on the next scan
    }
  })();
  return signedIn;
}

/** Forgets the session in this browser (after the server deleted the user) and starts a new one. */
export async function startNewIdentity(): Promise<void> {
  signedIn = null;
  try {
    await browserClient()?.auth.signOut({ scope: "local" });
  } catch {
    // the cookies are already cleared by the server
  }
  await ensureSignedIn();
}
