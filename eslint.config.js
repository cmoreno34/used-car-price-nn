/* One job: catch names that do not exist. Vite builds them without complaint
 * and the page goes blank in the browser. */
import globals from "globals";

export default [
  { ignores: ["dist/**", "node_modules/**"] },
  {
    files: ["src/**/*.{js,jsx}", "test/**/*.js"],
    languageOptions: { ecmaVersion: 2022, sourceType: "module", parserOptions: { ecmaFeatures: { jsx: true } }, globals: { ...globals.browser, ...globals.node } },
    rules: { "no-undef": "error", "no-unused-vars": ["warn", { args: "none", varsIgnorePattern: "^_|^React$" }] },
  },
];
