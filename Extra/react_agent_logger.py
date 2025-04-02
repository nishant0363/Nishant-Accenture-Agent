import re
import json
from datetime import datetime
from typing import Dict, List, Optional
import os

class ReActAgentLogger:
    def __init__(self, log_file: str = None, output_format: str = "markdown"):
        """
        Initialize the ReAct Agent Logger.
        
        Args:
            log_file: Path to save the log file. If None, a timestamped file will be created.
            output_format: Format to save logs ('markdown', 'json', or 'txt')
        """
        self.steps = []
        self.current_step = {}
        self.output_format = output_format
        
        # Create a timestamped log file if none provided
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = f"react_agent_log_{timestamp}.{self._get_extension()}"
        else:
            self.log_file = log_file
            
        # Initialize the log file with headers
        self._initialize_log_file()
    
    def _get_extension(self) -> str:
        """Get the appropriate file extension based on output format."""
        if self.output_format == "markdown":
            return "md"
        elif self.output_format == "json":
            return "json"
        else:
            return "txt"
    
    def _initialize_log_file(self):
        """Initialize the log file with appropriate headers."""
        if self.output_format == "markdown":
            with open(self.log_file, "w") as f:
                f.write(f"# ReAct Agent Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        elif self.output_format == "json":
            with open(self.log_file, "w") as f:
                f.write(json.dumps({"timestamp": datetime.now().isoformat(), "steps": []}, indent=2))
    
    def capture_output(self, output_line: str):
        """
        Process a line of output from the ReAct agent and extract relevant information.
        
        Args:
            output_line: A line of output from the ReAct agent
        """
        # Detect the start of a new step
        step_match = re.match(r"> Running step ([a-f0-9-]+)\. Step input: (.*)", output_line)
        if step_match:
            # If we have a current step in progress, save it
            if self.current_step:
                self.steps.append(self.current_step)
                self._write_step_to_log(self.current_step)
            
            # Start a new step
            step_id, step_input = step_match.groups()
            self.current_step = {
                "step_id": step_id,
                "input": None if step_input == "None" else step_input,
                "thought": None,
                "action": None,
                "action_input": None,
                "observation": None,
                "timestamp": datetime.now().isoformat()
            }
            return
            
        # Extract thought
        if output_line.startswith("Thought: "):
            self.current_step["thought"] = output_line[9:]
            
        # Extract action
        elif output_line.startswith("Action: "):
            self.current_step["action"] = output_line[8:]
            
        # Extract action input
        elif output_line.startswith("Action Input: "):
            # Try to parse as JSON if possible
            action_input_str = output_line[14:]
            try:
                self.current_step["action_input"] = json.loads(action_input_str.replace("'", '"'))
            except json.JSONDecodeError:
                self.current_step["action_input"] = action_input_str
                
        # Start of observation
        elif output_line.startswith("Observation: "):
            self.current_step["observation"] = output_line[13:]
            
        # Continuation of observation (for multi-line observations)
        elif self.current_step.get("observation") is not None and not any(
            output_line.startswith(prefix) for prefix in ["> Running step", "Thought:", "Action:", "Action Input:"]
        ):
            # Only append if it's not one of the markers for a new section
            self.current_step["observation"] += f"\n{output_line}"
            
        # Check for final answer
        elif output_line.startswith("Answer: "):
            if self.current_step:
                self.current_step["final_answer"] = output_line[8:]
                self.steps.append(self.current_step)
                self._write_step_to_log(self.current_step)
                self.current_step = {}
    
    def _write_step_to_log(self, step: Dict):
        """
        Write a step to the log file in the specified format.
        
        Args:
            step: Dictionary containing step information
        """
        if self.output_format == "markdown":
            with open(self.log_file, "a") as f:
                f.write(f"## Step: {step['step_id']}\n\n")
                if step["input"]:
                    f.write(f"**Input:** {step['input']}\n\n")
                if step["thought"]:
                    f.write(f"**Thought:** {step['thought']}\n\n")
                if step["action"]:
                    f.write(f"**Action:** {step['action']}\n\n")
                if step["action_input"]:
                    f.write(f"**Action Input:**\n```json\n{json.dumps(step['action_input'], indent=2)}\n```\n\n")
                if step["observation"]:
                    f.write(f"**Observation:**\n```\n{step['observation']}\n```\n\n")
                if step.get("final_answer"):
                    f.write(f"**Final Answer:** {step['final_answer']}\n\n")
                f.write("---\n\n")
                
        elif self.output_format == "json":
            with open(self.log_file, "r") as f:
                data = json.load(f)
            
            data["steps"].append(step)
            
            with open(self.log_file, "w") as f:
                f.write(json.dumps(data, indent=2))
                
        else:  # Plain text
            with open(self.log_file, "a") as f:
                f.write(f"=== STEP: {step['step_id']} ===\n")
                if step["input"]:
                    f.write(f"INPUT: {step['input']}\n")
                if step["thought"]:
                    f.write(f"THOUGHT: {step['thought']}\n")
                if step["action"]:
                    f.write(f"ACTION: {step['action']}\n")
                if step["action_input"]:
                    f.write(f"ACTION INPUT: {json.dumps(step['action_input'])}\n")
                if step["observation"]:
                    f.write(f"OBSERVATION:\n{step['observation']}\n")
                if step.get("final_answer"):
                    f.write(f"FINAL ANSWER: {step['final_answer']}\n")
                f.write("=" * 40 + "\n\n")
    
    def finalize(self):
        """Finalize the log file and return statistics."""
        # Save any pending step
        if self.current_step:
            self.steps.append(self.current_step)
            self._write_step_to_log(self.current_step)
            
        # Add summary statistics for markdown and text formats
        if self.output_format == "markdown":
            with open(self.log_file, "a") as f:
                f.write(f"# Summary\n\n")
                f.write(f"- Total Steps: {len(self.steps)}\n")
                f.write(f"- Actions Used: {', '.join(set(step['action'] for step in self.steps if step['action']))}\n")
                
        elif self.output_format == "json":
            with open(self.log_file, "r") as f:
                data = json.load(f)
            
            data["summary"] = {
                "total_steps": len(self.steps),
                "actions_used": list(set(step["action"] for step in self.steps if step["action"]))
            }
            
            with open(self.log_file, "w") as f:
                f.write(json.dumps(data, indent=2))
                
        else:  # Plain text
            with open(self.log_file, "a") as f:
                f.write("=== SUMMARY ===\n")
                f.write(f"Total Steps: {len(self.steps)}\n")
                f.write(f"Actions Used: {', '.join(set(step['action'] for step in self.steps if step['action']))}\n")
                
        return {
            "log_file": self.log_file,
            "total_steps": len(self.steps)
        }


# Example usage with your agent setup
def setup_react_agent_with_logger(output_format="markdown"):
    """
    Set up a ReAct agent with logging capabilities.
    
    Args:
        output_format: Format to save the logs ('markdown', 'json', or 'txt')
        
    Returns:
        Tuple of (agent, logger)
    """
    # Initialize your agent as you normally would
    # agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)
    
    # Initialize the logger
    logger = ReActAgentLogger(output_format=output_format)
    
    return logger

# Modified agent query loop with logging
def run_agent_with_logging(agent, logger):
    """
    Run the agent with logging of all steps.
    
    Args:
        agent: The ReAct agent
        logger: The ReActAgentLogger instance
    """
    import io
    import sys
    
    class OutputCapture:
        def __init__(self, logger):
            self.logger = logger
            self.original_stdout = sys.stdout
            self.buffer = ""
        
        def write(self, text):
            self.original_stdout.write(text)
            self.buffer += text
            if '\n' in text:
                lines = self.buffer.split('\n')
                for line in lines[:-1]:  # Process all complete lines
                    self.logger.capture_output(line)
                self.buffer = lines[-1]  # Keep any partial line
        
        def flush(self):
            self.original_stdout.flush()
    
    # Replace stdout with our capturing stdout
    capture = OutputCapture(logger)
    sys.stdout = capture
    
    try:
        while (prompt := input("Enter a prompt (q to quit): ")) != "q":
            result = agent.query(prompt)
            print(result)
    finally:
        # Restore stdout
        sys.stdout = capture.original_stdout
        
        # Finalize the log
        summary = logger.finalize()
        print(f"\nLogging complete. Log saved to: {summary['log_file']}")
        print(f"Total steps recorded: {summary['total_steps']}")


# Example of how to use it
if __name__ == "__main__":
    # This is how you would implement it in your code
    '''
    from llama_index.core.agent import ReActAgent
    
    # Set up your agent
    agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)
    
    # Set up the logger
    logger = setup_react_agent_with_logger(output_format="markdown")
    
    # Run the agent with logging
    run_agent_with_logging(agent, logger)
    '''
    
    # For testing/demo without actual agent
    class DummyAgent:
        def query(self, prompt):
            print(f"> Running step 1234-5678. Step input: {prompt}")
            print("Thought: I need to analyze this.")
            print("Action: some_tool")
            print("Action Input: {'input': 'test'}")
            print("Observation: Result of analysis")
            print("Answer: This is the answer")
            return "This is the answer"
    
    # Demo with dummy agent
    agent = DummyAgent()
    logger = setup_react_agent_with_logger()
    
    # Manually simulate running with captured output
    logger.capture_output("> Running step 1234-5678. Step input: test query")
    logger.capture_output("Thought: I need to analyze this.")
    logger.capture_output("Action: some_tool")
    logger.capture_output("Action Input: {'input': 'test'}")
    logger.capture_output("Observation: Result of analysis")
    logger.capture_output("Answer: This is the answer")
    
    # Finalize the log
    summary = logger.finalize()
    print(f"Demo log saved to: {summary['log_file']}")