export const base = import.meta.env.VITE_API_URL || "";
export const headers = () => ({
  ...(sessionStorage.getItem("aurora-token")
    ? { Authorization: `Bearer ${sessionStorage.getItem("aurora-token")}` }
    : {}),
});
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(base + path, {
    ...init,
    headers: { ...headers(), ...init.headers },
  });
  const body = await response.json();
  if (!response.ok)
    throw new Error(
      body.error?.message || `Request gagal (${response.status})`,
    );
  return body as T;
}
export async function download(path: string, filename: string) {
  const response = await fetch(base + path, { headers: headers() });
  if (!response.ok) {
    const body = await response.json();
    throw new Error(body.error?.message || "Unduhan gagal");
  }
  const uri = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = uri;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(uri), 5000);
}
