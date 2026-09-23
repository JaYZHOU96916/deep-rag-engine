const path = require("node:path");
const { execFileSync } = require("node:child_process");

module.exports = {
  packagerConfig: {
    name: "Deep-RAG",
    appBundleId: "com.jayzhou.deeprag",
    asar: true,
    extraResource: [
      path.resolve(__dirname, "../backend/dist/deep-rag-api"),
      path.resolve(__dirname, "../frontend/out"),
    ],
  },
  makers: [
    {
      name: "@electron-forge/maker-dmg",
      config: { name: "Deep-RAG" },
    },
  ],
  hooks: {
    postPackage: async (_config, result) => {
      if (result.platform !== "darwin") return;
      for (const outputPath of result.outputPaths) {
        const appPath = path.join(outputPath, "Deep-RAG.app");
        execFileSync("/usr/bin/codesign", ["--force", "--deep", "--sign", "-", appPath], {
          stdio: "inherit",
        });
        execFileSync("/usr/bin/codesign", ["--verify", "--deep", "--strict", appPath], {
          stdio: "inherit",
        });
      }
    },
  },
};
