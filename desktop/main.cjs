const { app, BrowserWindow, dialog, ipcMain, shell } = require("electron");
const { spawn } = require("node:child_process");
const { randomBytes } = require("node:crypto");
const http = require("node:http");
const net = require("node:net");
const path = require("node:path");

let window;
let backend;
let appUrl;
const sessionToken = randomBytes(32).toString("hex");

ipcMain.on("desktop-session-token", (event) => {
  event.returnValue = sessionToken;
});

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close(() => resolve(address.port));
    });
  });
}

function healthy(url) {
  return new Promise((resolve) => {
    const request = http.get(`${url}/healthz`, { timeout: 1000 }, (response) => {
      response.resume();
      resolve(response.statusCode === 200);
    });
    request.once("timeout", () => { request.destroy(); resolve(false); });
    request.once("error", () => resolve(false));
  });
}

async function waitForBackend(url) {
  for (let attempt = 0; attempt < 480; attempt += 1) {
    if (backend.exitCode !== null) throw new Error("本机服务未能启动。");
    if (await healthy(url)) return;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("本机服务启动超时。请重新打开 Deep-RAG。");
}

function backendCommand() {
  if (app.isPackaged) {
    return { executable: path.join(process.resourcesPath, "deep-rag-api", "deep-rag-api"), args: [] };
  }
  return {
    executable: process.env.DEEP_RAG_DESKTOP_PYTHON || path.resolve(__dirname, "../.venv/bin/python"),
    args: ["-m", "desktop_api.server"],
  };
}

function frontendDirectory() {
  return app.isPackaged
    ? path.join(process.resourcesPath, "out")
    : path.resolve(__dirname, "../frontend/out");
}

function openExternal(url) {
  try {
    const parsed = new URL(url);
    if (parsed.protocol === "https:" || parsed.protocol === "http:") void shell.openExternal(url);
  } catch {
    // Only valid web addresses may leave the desktop window.
  }
}

async function start() {
  const port = await freePort();
  appUrl = `http://127.0.0.1:${port}`;
  const command = backendCommand();
  backend = spawn(command.executable, command.args, {
    cwd: app.isPackaged ? process.resourcesPath : path.resolve(__dirname, "../backend"),
    env: {
      ...process.env,
      PYTHONPATH: app.isPackaged ? undefined : path.resolve(__dirname, "../backend"),
      DEEP_RAG_DESKTOP_PORT: String(port),
      DEEP_RAG_DESKTOP_DATA_DIR: path.join(app.getPath("userData"), "data"),
      DEEP_RAG_DESKTOP_FRONTEND_DIR: frontendDirectory(),
      DEEP_RAG_DESKTOP_SESSION_TOKEN: sessionToken,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  backend.stdout.on("data", (chunk) => process.stdout.write(chunk));
  backend.stderr.on("data", (chunk) => process.stderr.write(chunk));
  await waitForBackend(appUrl);

  window = new BrowserWindow({
    width: 1440,
    height: 950,
    minWidth: 980,
    minHeight: 650,
    show: false,
    title: "Deep-RAG",
    backgroundColor: "#F7FAFC",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
    },
  });
  window.webContents.setWindowOpenHandler(({ url }) => {
    openExternal(url);
    return { action: "deny" };
  });
  window.webContents.on("will-navigate", (event, url) => {
    if (!url.startsWith(`${appUrl}/`)) {
      event.preventDefault();
      openExternal(url);
    }
  });
  window.once("ready-to-show", () => window.show());
  await window.loadURL(appUrl);
}

app.whenReady().then(async () => {
  try {
    await start();
  } catch (error) {
    dialog.showErrorBox("Deep-RAG 无法启动", String(error.message || error));
    app.quit();
  }
});

app.on("activate", () => {
  if (window && !window.isDestroyed()) window.show();
});

app.on("before-quit", () => {
  if (backend && backend.exitCode === null) backend.kill("SIGTERM");
});
