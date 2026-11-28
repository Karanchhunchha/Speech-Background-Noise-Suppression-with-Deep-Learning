function plan = buildfile
%BUILDFILE Build plan for this project (MATLAB build tool, R2022b+).
%   Run `buildtool` in this folder to execute the default task
%   (verify: code check + full test suite + submission verification),
%   `buildtool test` for the test suite only, `buildtool -tasks` to
%   list every task. This is the same entry point CI uses.

    import matlab.buildtool.tasks.CodeIssuesTask
    import matlab.buildtool.tasks.TestTask
    import matlab.buildtool.tasks.CleanTask

    plan = buildplan(localfunctions);
    plan("check") = CodeIssuesTask;
    plan("test") = TestTask;
    plan("clean") = CleanTask;
    plan("verify").Dependencies = ["check" "test"];
    plan.DefaultTasks = "verify";
end

function verifyTask(~)
%VERIFY Custom task: the project's own submission gate.
    assert(verify_submission(), ...
        'verify_submission reported a failed check.');
end
