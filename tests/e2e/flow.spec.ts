import { expect, test } from "../../frontend/node_modules/@playwright/test";

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
