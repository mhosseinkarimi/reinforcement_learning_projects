document$.subscribe(() => {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "\\[", right: "\\]", display: true  },
      { left: "\\(", right: "\\)", display: false },
      { left: "$$",  right: "$$", display: true  },
      { left: "$",   right: "$",   display: false }
    ],
    ignoredTags: ["script","noscript","style","textarea","pre","code"]
  });
});
