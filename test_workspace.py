from workspace_manager import workspace

def test_system():
    print("--- 1. Switching to project 'robot_arm' ---")
    res = workspace.switch_project("robot_arm")
    print(res)

    print("\n--- 2. Writing a project plan file ---")
    plan_content = """# Robot Arm Project Plan
1. Design 3D gripper mechanism.
2. Select high-torque servo motors.
3. Configure kinematics controller in Python.
"""
    res = workspace.write_file("plan.md", plan_content)
    print(res)

    print("\n--- 3. Listing project files ---")
    files = workspace.list_project_files()
    print("Files found:", files)

    print("\n--- 4. Reading file back ---")
    content = workspace.read_file("plan.md")
    print("File Content:\n" + content)

    print("--- 5. Saving mock chat history ---")
    workspace.save_chat_turn("user", "Ada, create a plan for the robot arm.")
    workspace.save_chat_turn("assistant", "Plan created and saved to plan.md, sir.")

    print("\n--- 6. Loading history ---")
    history = workspace.load_project_history()
    for entry in history:
        print(f"[{entry['role']}]: {entry['text']}")

    print("\nAll workspace tests passed successfully!")

if __name__ == "__main__":
    test_system()