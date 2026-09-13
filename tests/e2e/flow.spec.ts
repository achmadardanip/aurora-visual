import {
  expect,
  test,
  type Page,
} from "../../frontend/node_modules/@playwright/test";

async function solidPng(page: Page, hex: string, size = 128): Promise<Buffer> {
  // Generate a solid-color PNG inside the browser via canvas; Playwright
  // returns it as base64 which we decode into a Node buffer for setInputFiles.
  const dataUrl = await page.evaluate(
    ([color, px]) => {
      const canvas = document.createElement("canvas");
      canvas.width = px;
      canvas.height = px;
      const ctx = canvas.getContext("2d")!;
      ctx.fillStyle = color;
      ctx.fillRect(0, 0, px, px);
      return canvas.toDataURL("image/png");
    },
    [hex, size] as [string, number],
  );
  return Buffer.from(dataUrl.split(",")[1], "base64");
}

test("demo contradiction, atom correction, reanalysis and history", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Periksa klaim, satu per satu." }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Bantahan", exact: true }).click();
  await page
    .getByRole("button", { name: "Jalankan analisis", exact: true })
    .click();
  await expect(
    page.getByRole("button", {
      name: /Bidang ini berwarna merah Bertentangan/,
    }),
  ).toBeVisible({ timeout: 30000 });
  await expect(page.getByText("TAHAP 1 · SCREENING ASAL MEDIA")).toBeVisible();
  await expect(
    page.getByText(
      "Screening asal media tidak menentukan kebenaran caption dan tidak mengubah status Didukung, Bertentangan, atau Tidak teramati.",
    ),
  ).toBeVisible();
  await expect(page.getByText("Tahap 3 · multimodal")).toBeVisible();
  await expect(page.locator(".region.contra")).toHaveCount(16);
  await page.getByRole("button", { name: "Koreksi atom" }).click();
  await page
    .getByLabel("Proposisi", { exact: true })
    .fill("Bidang ini berwarna biru");
  await page.getByLabel("Objek", { exact: true }).fill("biru");
  await page
    .getByLabel("Alasan koreksi")
    .fill("E2E: koreksi atom warna untuk memeriksa invalidasi hasil.");
  await page.getByRole("button", { name: "Simpan koreksi" }).click();
  await expect(
    page.getByText("Menunggu analisis ulang", { exact: true }),
  ).toBeVisible();
  await expect(page.locator("#caption")).toHaveValue(
    "Bidang ini berwarna merah",
  );
  await page
    .getByRole("button", { name: "Analisis ulang", exact: true })
    .click();
  await expect(
    page.getByRole("button", {
      name: /Bidang ini berwarna biru Didukung visual/,
    }),
  ).toBeVisible({ timeout: 30000 });
  await page.reload();
  await expect(
    page.getByRole("button", {
      name: /Bidang ini berwarna biru Didukung visual/,
    }),
  ).toBeVisible();
  await page.screenshot({
    path: "../artifacts/reports/browser-desktop.png",
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "Riwayat kasus", exact: true })
    .click();
  await expect(page.locator(".history-row").first()).toBeVisible();
});

test("mobile empty state and unobservable event claims", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Jalankan analisis", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "Tak teramati", exact: true }).click();
  await page
    .getByRole("button", { name: "Jalankan analisis", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: /September 2026 Tidak teramati/ }),
  ).toBeVisible({ timeout: 30000 });
  await expect(page.getByText("TAHAP 1 · SCREENING ASAL MEDIA")).toBeVisible();
  const detectorRow = page.locator(".screening-ledger > div", {
    hasText: "Detektor AI / deepfake",
  });
  await expect(detectorRow.getByText("Belum dikonfigurasi")).toBeVisible();
  await expect(
    detectorRow.getByText(/bukan verdict autentisitas/),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /di Monas Tidak teramati/ }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () =>
        document.documentElement.scrollWidth <=
        document.documentElement.clientWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: "../artifacts/reports/browser-mobile.png",
    fullPage: true,
  });
});

test("multi-image upload: validation, preview, removal and analysis", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Live", exact: true }).click();

  const redA = await solidPng(page, "#e02222");
  const redB = await solidPng(page, "#d91a1a");
  const blue = await solidPng(page, "#2255dd");

  // Invalid type is rejected client-side with a clear message.
  await page
    .locator('input[type="file"][aria-label="Unggah gambar"]')
    .setInputFiles([
      { name: "merah-a.png", mimeType: "image/png", buffer: redA },
      {
        name: "catatan.txt",
        mimeType: "text/plain",
        buffer: Buffer.from("bukan gambar"),
      },
      { name: "merah-b.png", mimeType: "image/png", buffer: redB },
    ]);
  await expect(
    page.getByText(/catatan\.txt: format text\/plain tidak didukung/),
  ).toBeVisible();
  await expect(page.locator(".preview-strip .thumb-item")).toHaveCount(2);
  await expect(
    page.getByRole("button", { name: "Unggah 2 gambar" }),
  ).toBeVisible();

  // Unwanted preview can be removed before upload.
  await page
    .getByRole("button", { name: "Hapus pratinjau merah-b.png" })
    .click();
  await expect(page.locator(".preview-strip .thumb-item")).toHaveCount(1);

  await page
    .getByRole("button", { name: "Unggah 1 gambar", exact: true })
    .click();
  await expect(
    page.getByText("1 gambar berhasil diunggah dan siap dianalisis."),
  ).toBeVisible();
  await expect(page.locator(".thumb-strip .thumb-item")).toHaveCount(1);
  await expect(page.getByText("1/8 terunggah")).toBeVisible();

  // A second batch can be added up to the limit.
  await page
    .locator('input[type="file"][aria-label="Unggah gambar"]')
    .setInputFiles([{ name: "biru.png", mimeType: "image/png", buffer: blue }]);
  await page
    .getByRole("button", { name: "Unggah 1 gambar", exact: true })
    .click();
  await expect(page.locator(".thumb-strip .thumb-item")).toHaveCount(2);
  await expect(page.getByText("2/8 terunggah")).toBeVisible();

  // Uploaded image can be dropped from the selection.
  await page.locator(".thumb-strip .thumb-remove").nth(1).click();
  await expect(page.locator(".thumb-strip .thumb-item")).toHaveCount(1);
  await expect(page.getByText("1/8 terunggah")).toBeVisible();

  // Red field claim on the remaining red image is supported end to end.
  await page.locator("#caption").fill("Bidang ini berwarna merah");
  await page
    .getByRole("button", { name: "Jalankan analisis", exact: true })
    .click();
  await expect(
    page.getByRole("button", {
      name: /Bidang ini berwarna merah Didukung visual/,
    }),
  ).toBeVisible({ timeout: 30000 });
  await expect(page.locator(".evidence-card .image-frame img")).toHaveCount(1);
  await expect(page.locator(".region.support")).toHaveCount(16);
});
