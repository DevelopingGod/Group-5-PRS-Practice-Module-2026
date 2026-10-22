import { cookies } from "next/headers";

import { currentUserId, deleteUser } from "@/lib/supabase/server";

// "Reset my identity" on the profile page: deletes the user's pseudonym, points and scan
// history, then clears the session cookies so the next scan starts a new pseudonymous user.
export async function POST(request: Request) {
  // Only the app itself may ask (cookies are SameSite=Lax, this is a second guard).
  const origin = request.headers.get("origin");
  if (origin && new URL(origin).host !== new URL(request.url).host) {
    return Response.json({ code: "invalid_state", message: "Not allowed." }, { status: 403 });
  }
  try {
    const userId = await currentUserId();
    if (userId) await deleteUser(userId);
  } catch (err) {
    console.error("[reset-identity]", err);
    return Response.json(
      { code: "internal", message: "We could not reset your identity. Please try again." },
      { status: 500 },
    );
  }
  const store = await cookies();
  for (const { name } of store.getAll()) if (name.startsWith("sb-")) store.delete(name);
  return new Response(null, { status: 204 });
}
