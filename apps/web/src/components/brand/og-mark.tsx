// The Bin Buddy mark for generated images (app icon, link preview). ImageResponse renders plain
// JSX with inline styles only, so this mirrors BinMark in icons.tsx on a bin-blue tile.
/** `rounded={false}` for the iPhone icon: iOS rounds the corners itself. */
export function OgMark({ size, rounded = true }: { size: number; rounded?: boolean }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: size,
        height: size,
        borderRadius: rounded ? size / 4 : 0,
        background: "#1F5FD1",
      }}
    >
      <svg
        width={size * 0.75}
        height={size * 0.75}
        viewBox="0 0 24 24"
        fill="none"
        stroke="#F4F7FB"
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M5 7.5 18.6 4.6" />
        <path d="M10 5.5 9.7 4a1 1 0 0 1 .8-1.2l2.3-.5a1 1 0 0 1 1.2.8l.3 1.4" />
        <path d="M6 10h12l-1.1 10.1a1.5 1.5 0 0 1-1.5 1.4H8.6a1.5 1.5 0 0 1-1.5-1.4Z" />
        <path d="M10 13.5v4.5M14 13.5v4.5" />
      </svg>
    </div>
  );
}
