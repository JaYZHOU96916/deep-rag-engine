const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("deepRagDesktop", {
  sessionToken: ipcRenderer.sendSync("desktop-session-token"),
});
