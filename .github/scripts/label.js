const { Octokit } = require("@octokit/rest");
const { context } = require("@actions/github");

const octokit = new Octokit({ auth: process.env.GITHUB_TOKEN });

const run = async () => {
  const payload = context.payload;
  const isIssue = !!payload.issue;
  const title = isIssue ? payload.issue.title : payload.pull_request.title;
  const number = isIssue ? payload.issue.number : payload.pull_request.number;
  const labels = [];

  console.log(`Event type: ${isIssue ? "Issue" : "Pull Request"}`);
  console.log(`Title: ${title}`);
  console.log(`Number: ${number}`);

  // 获取仓库中的所有标签
  const repoLabels = await octokit.paginate(octokit.issues.listLabelsForRepo, {
    owner: context.repo.owner,
    repo: context.repo.repo,
  });

  console.log("Repository labels:", repoLabels);

  // 遍历仓库中的标签，根据标题添加标签
  for (const label of repoLabels) {
    if (title.toLowerCase().includes(label.name.toLowerCase())) {
      labels.push(label.name);
    }
  }

  console.log("Labels to add:", labels);

  if (labels.length > 0) {
    await octokit.issues.addLabels({
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: number,
      labels: labels,
    });
    console.log("Labels added successfully");
  } else {
    console.log("No labels to add");
  }
};

run().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});