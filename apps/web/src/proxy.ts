// Keeps the anonymous Supabase session fresh before a page renders. Server Components cannot set
// cookies, so without this a page rendered after the access token expired (one hour) would treat
// the user as signed out: no points in the header, no "(you)" on the leaderboard, an empty
// profile. The refresh keeps the same user id, so the pseudonym is kept.
import { createServerClient } from "@supabase/ssr";
import { NextResponse, type NextRequest } from "next/server";

export async function proxy(request: NextRequest) {
  let response = NextResponse.next({ request });
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  const hasSession = request.cookies.getAll().some((c) => c.name.startsWith("sb-"));
  if (!url || !key || !hasSession) return response;

  const supabase = createServerClient(url, key, {
    cookies: {
      getAll: () => request.cookies.getAll(),
      setAll: (toSet, headers) => {
        // The page reads the request cookies; the browser keeps the response cookies.
        for (const { name, value } of toSet) request.cookies.set(name, value);
        response = NextResponse.next({ request });
        for (const { name, value, options } of toSet) response.cookies.set(name, value, options);
        for (const [name, value] of Object.entries(headers)) response.headers.set(name, value);
      },
    },
  });
  // Reads the session from the cookie and refreshes it only if the access token has expired,
  // so most requests make no network call. The page still validates the user with getUser().
  await supabase.auth.getSession();
  return response;
}

export const config = {
  // Pages only: route handlers set cookies themselves, and static files need no session.
  matcher: ["/((?!api/|_next/static|_next/image|favicon.ico|.*\\.[a-z0-9]+$).*)"],
};
