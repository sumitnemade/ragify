#!/usr/bin/env python3
"""
RAGify Test Runner

A comprehensive command-line test runner for the RAGify framework that allows
for categorized test execution, performance monitoring, and report generation.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Test categories and their corresponding pytest markers
TEST_CATEGORIES = {
    "unit": "unit",
    "integration": "integration", 
    "performance": "performance",
    "security": "security",
    "load": "load",
    "memory": "memory",
    "regression": "regression",
    "vulnerability": "vulnerability",
    "configuration": "configuration"
}

# Test markers for different types of tests
TEST_MARKERS = {
    "unit": "unit",
    "integration": "integration",
    "performance": "performance", 
    "security": "security",
    "load": "load",
    "memory": "memory",
    "slow": "slow",
    "network": "network",
    "database": "database",
    "vulnerability": "vulnerability",
    "configuration": "configuration",
    "regression": "regression"
}

# Performance thresholds for different test categories
PERFORMANCE_THRESHOLDS = {
    "unit": {"max_time": 1.0, "max_memory": 100},
    "integration": {"max_time": 5.0, "max_memory": 200},
    "performance": {"max_time": 30.0, "max_memory": 500},
    "security": {"max_time": 10.0, "max_memory": 150},
    "load": {"max_time": 60.0, "max_memory": 1000}
}


class RAGifyTestRunner:
    """Main test runner class for RAGify framework."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.results_dir = self.project_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)
        
    def run_tests(self, 
                  categories: Optional[List[str]] = None,
                  markers: Optional[List[str]] = None,
                  verbose: bool = False,
                  coverage: bool = True,
                  parallel: bool = False,
                  output_format: str = "text") -> Dict:
        """Run tests with specified parameters."""
        
        # Build pytest arguments
        pytest_args = self._build_pytest_args(
            categories=categories,
            markers=markers,
            verbose=verbose,
            coverage=coverage,
            parallel=parallel
        )
        
        print(f"Running tests with args: {' '.join(pytest_args)}")
        print(f"Project root: {self.project_root}")
        print(f"Results directory: {self.results_dir}")
        print("-" * 60)
        
        # Run tests
        start_time = time.time()
        result = self._execute_pytest(pytest_args)
        end_time = time.time()
        
        # Collect and process results
        test_results = self._collect_results(result, end_time - start_time)
        
        # Generate report
        report = self._generate_report(test_results, output_format)
        
        # Save results
        self._save_results(test_results, report)
        
        return test_results
    
    def _build_pytest_args(self, 
                           categories: Optional[List[str]] = None,
                           markers: Optional[List[str]] = None,
                           verbose: bool = False,
                           coverage: bool = True,
                           parallel: bool = False) -> List[str]:
        """Build pytest command line arguments."""
        
        args = ["python", "-m", "pytest"]
        
        # Add test directory
        args.append("tests/")
        
        # Add categories
        if categories:
            for category in categories:
                if category in TEST_CATEGORIES:
                    args.extend(["-m", TEST_CATEGORIES[category]])
        
        # Add markers
        if markers:
            for marker in markers:
                if marker in TEST_MARKERS:
                    args.extend(["-m", TEST_MARKERS[marker]])
        
        # Add coverage
        if coverage:
            args.extend([
                "--cov=src/ragify",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing"
            ])
        
        # Add parallel execution
        if parallel:
            args.extend(["-n", "auto"])
        
        # Add verbosity
        if verbose:
            args.extend(["-v", "--tb=long"])
        else:
            args.extend(["--tb=short"])
        
        # Add other useful options
        args.extend([
            "--strict-markers",
            "--disable-warnings",
            "--durations=10"
        ])
        
        return args
    
    def _execute_pytest(self, args: List[str]) -> subprocess.CompletedProcess:
        """Execute pytest with given arguments."""
        
        try:
            result = subprocess.run(
                args,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            return result
        except subprocess.TimeoutExpired:
            print("Test execution timed out after 1 hour")
            sys.exit(1)
        except Exception as e:
            print(f"Error running tests: {e}")
            sys.exit(1)
    
    def _collect_results(self, result: subprocess.CompletedProcess, duration: float) -> Dict:
        """Collect and parse test results."""
        
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "summary": self._parse_test_summary(result.stdout),
            "coverage": self._parse_coverage(result.stdout),
            "performance": self._extract_performance_metrics(result.stdout)
        }
        
        return test_results
    
    def _parse_test_summary(self, stdout: str) -> Dict:
        """Parse test execution summary from pytest output."""
        
        summary = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "xfailed": 0,
            "xpassed": 0
        }
        
        # Look for summary line like: "==== 295 passed, 4 skipped, 1 xpassed, 26 warnings ===="
        for line in stdout.split('\n'):
            if "====" in line and ("passed" in line or "failed" in line):
                # Parse the summary line
                parts = line.replace("====", "").strip().split(',')
                for part in parts:
                    part = part.strip()
                    if "passed" in part and "xpassed" not in part:
                        summary["passed"] = int(part.split()[0])
                    elif "failed" in part:
                        summary["failed"] = int(part.split()[0])
                    elif "skipped" in part:
                        summary["skipped"] = int(part.split()[0])
                    elif "errors" in part:
                        summary["errors"] = int(part.split()[0])
                    elif "xfailed" in part:
                        summary["xfailed"] = int(part.split()[0])
                    elif "xpassed" in part:
                        summary["xpassed"] = int(part.split()[0])
                
                summary["total_tests"] = sum([
                    summary["passed"], summary["failed"], summary["skipped"], 
                    summary["errors"], summary["xfailed"], summary["xpassed"]
                ])
                break
        
        return summary
    
    def _parse_coverage(self, stdout: str) -> Dict:
        """Parse coverage information from pytest output."""
        
        coverage = {
            "total_coverage": 0.0,
            "modules": {}
        }
        
        in_coverage = False
        for line in stdout.split('\n'):
            if "---------- coverage:" in line:
                in_coverage = True
                continue
            
            if in_coverage and line.strip() == "":
                in_coverage = False
                continue
            
            if in_coverage and "TOTAL" in line:
                # Parse total coverage line
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        coverage["total_coverage"] = float(parts[-1].replace('%', ''))
                    except ValueError:
                        pass
                break
            
            if in_coverage and "src/ragify" in line:
                # Parse module coverage
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        module_name = parts[0].replace("src/ragify/", "")
                        module_coverage = float(parts[-1].replace('%', ''))
                        coverage["modules"][module_name] = module_coverage
                    except ValueError:
                        pass
        
        return coverage
    
    def _extract_performance_metrics(self, stdout: str) -> Dict:
        """Extract performance metrics from test output."""
        
        metrics = {
            "slowest_tests": [],
            "memory_usage": {},
            "execution_times": {}
        }
        
        # Look for duration information
        for line in stdout.split('\n'):
            if "slowest durations" in line:
                # Parse slowest test durations
                continue
            elif line.strip().startswith("test_"):
                # This might be a test duration line
                if "s" in line and "test_" in line:
                    try:
                        parts = line.split()
                        test_name = parts[0]
                        duration_str = parts[-1]
                        if duration_str.endswith('s'):
                            duration = float(duration_str[:-1])
                            metrics["execution_times"][test_name] = duration
                    except (ValueError, IndexError):
                        pass
        
        return metrics
    
    def _generate_report(self, test_results: Dict, output_format: str) -> str:
        """Generate test execution report."""
        
        if output_format == "json":
            return json.dumps(test_results, indent=2)
        
        # Text format report
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("RAGify Test Execution Report")
        report_lines.append("=" * 60)
        report_lines.append(f"Timestamp: {test_results['timestamp']}")
        report_lines.append(f"Duration: {test_results['duration']:.2f} seconds")
        report_lines.append(f"Return Code: {test_results['return_code']}")
        report_lines.append("")
        
        # Test Summary
        summary = test_results['summary']
        report_lines.append("Test Summary:")
        report_lines.append(f"  Total Tests: {summary['total_tests']}")
        report_lines.append(f"  Passed: {summary['passed']}")
        report_lines.append(f"  Failed: {summary['failed']}")
        report_lines.append(f"  Skipped: {summary['skipped']}")
        report_lines.append(f"  Errors: {summary['errors']}")
        report_lines.append(f"  XFailed: {summary['xfailed']}")
        report_lines.append(f"  XPassed: {summary['xpassed']}")
        report_lines.append("")
        
        # Coverage Summary
        coverage = test_results['coverage']
        report_lines.append("Coverage Summary:")
        report_lines.append(f"  Total Coverage: {coverage['total_coverage']:.1f}%")
        report_lines.append("  Module Coverage:")
        for module, cov in sorted(coverage['modules'].items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"    {module}: {cov:.1f}%")
        report_lines.append("")
        
        # Performance Summary
        performance = test_results['performance']
        if performance['execution_times']:
            report_lines.append("Performance Summary:")
            slowest_tests = sorted(
                performance['execution_times'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            for test_name, duration in slowest_tests:
                report_lines.append(f"  {test_name}: {duration:.2f}s")
            report_lines.append("")
        
        # Status
        if test_results['return_code'] == 0:
            report_lines.append("✅ All tests passed successfully!")
        else:
            report_lines.append("❌ Some tests failed. Check output for details.")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def _save_results(self, test_results: Dict, report: str):
        """Save test results and report to files."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = self.results_dir / f"test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Save report as text
        report_file = self.results_dir / f"test_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Results saved to: {results_file}")
        print(f"Report saved to: {report_file}")


class PerformanceTestRunner(RAGifyTestRunner):
    """Specialized runner for performance tests."""
    
    def run_performance_tests(self, 
                             iterations: int = 3,
                             benchmark: bool = True,
                             memory_profiling: bool = True) -> Dict:
        """Run performance tests with specific configurations."""
        
        print("Running Performance Tests...")
        print(f"Iterations: {iterations}")
        print(f"Benchmark: {benchmark}")
        print(f"Memory Profiling: {memory_profiling}")
        
        # Add performance-specific markers
        markers = ["performance", "benchmark"] if benchmark else ["performance"]
        
        # Run tests multiple times for performance analysis
        all_results = []
        for i in range(iterations):
            print(f"\n--- Performance Test Run {i+1}/{iterations} ---")
            results = self.run_tests(
                categories=["performance"],
                markers=markers,
                verbose=False,
                coverage=False
            )
            all_results.append(results)
        
        # Aggregate performance results
        aggregated_results = self._aggregate_performance_results(all_results)
        
        # Add the standard structure expected by the main function
        aggregated_results['return_code'] = 0 if all(r['return_code'] == 0 for r in all_results) else 1
        aggregated_results['duration'] = aggregated_results['average_duration']
        
        return aggregated_results
    
    def _aggregate_performance_results(self, results: List[Dict]) -> Dict:
        """Aggregate results from multiple performance test runs."""
        
        aggregated = {
            "total_runs": len(results),
            "average_duration": sum(r['duration'] for r in results) / len(results),
            "min_duration": min(r['duration'] for r in results),
            "max_duration": max(r['duration'] for r in results),
            "success_rate": sum(1 for r in results if r['return_code'] == 0) / len(results),
            "detailed_results": results
        }
        
        return aggregated


class SecurityTestRunner(RAGifyTestRunner):
    """Specialized runner for security tests."""
    
    def run_security_tests(self, 
                          vulnerability_scan: bool = True,
                          compliance_check: bool = True) -> Dict:
        """Run security tests with specific configurations."""
        
        print("Running Security Tests...")
        print(f"Vulnerability Scan: {vulnerability_scan}")
        print(f"Compliance Check: {compliance_check}")
        
        # Add security-specific markers
        markers = ["security"]
        if vulnerability_scan:
            markers.append("vulnerability")
        if compliance_check:
            markers.append("configuration")
        
        # Run security tests
        results = self.run_tests(
            categories=["security"],
            markers=markers,
            verbose=True,
            coverage=True
        )
        
        # Add security-specific analysis
        security_analysis = self._analyze_security_results(results)
        results['security_analysis'] = security_analysis
        
        return results
    
    def _analyze_security_results(self, results: Dict) -> Dict:
        """Analyze security test results for vulnerabilities."""
        
        analysis = {
            "vulnerabilities_found": 0,
            "compliance_issues": 0,
            "security_score": 100,
            "recommendations": []
        }
        
        # Analyze test results for security issues
        if results['return_code'] != 0:
            analysis['security_score'] -= 20
            analysis['recommendations'].append("Fix failing security tests")
        
        # Check for specific security markers in output
        stdout = results['stdout'].lower()
        if "vulnerability" in stdout:
            analysis['vulnerabilities_found'] += 1
            analysis['security_score'] -= 30
            analysis['recommendations'].append("Address identified vulnerabilities")
        
        if "compliance" in stdout and "fail" in stdout:
            analysis['compliance_issues'] += 1
            analysis['security_score'] -= 25
            analysis['recommendations'].append("Fix compliance issues")
        
        # Ensure security score doesn't go below 0
        analysis['security_score'] = max(0, analysis['security_score'])
        
        return analysis


def main():
    """Main entry point for the test runner."""
    
    parser = argparse.ArgumentParser(
        description="RAGify Test Runner - Comprehensive testing for RAGify framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python scripts/run_tests.py
  
  # Run only unit tests
  python scripts/run_tests.py --categories unit
  
  # Run performance tests with 5 iterations
  python scripts/run_tests.py --performance --iterations 5
  
  # Run security tests with verbose output
  python scripts/run_tests.py --security --verbose
  
  # Run specific test markers
  python scripts/run_tests.py --markers slow,network
  
  # Generate JSON report
  python scripts/run_tests.py --output-format json
        """
    )
    
    # Test selection options
    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        choices=list(TEST_CATEGORIES.keys()),
        help="Test categories to run"
    )
    
    parser.add_argument(
        "--markers", "-m",
        nargs="+",
        choices=list(TEST_MARKERS.keys()),
        help="Test markers to include"
    )
    
    # Specialized test runners
    parser.add_argument(
        "--performance", "-p",
        action="store_true",
        help="Run performance tests with specialized runner"
    )
    
    parser.add_argument(
        "--security", "-s",
        action="store_true",
        help="Run security tests with specialized runner"
    )
    
    # Performance test options
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations for performance tests (default: 3)"
    )
    
    # Execution options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format for reports (default: text)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create appropriate test runner
    if args.performance:
        runner = PerformanceTestRunner()
        results = runner.run_performance_tests(
            iterations=args.iterations,
            benchmark=True,
            memory_profiling=True
        )
    elif args.security:
        runner = SecurityTestRunner()
        results = runner.run_security_tests(
            vulnerability_scan=True,
            compliance_check=True
        )
    else:
        runner = RAGifyTestRunner()
        results = runner.run_tests(
            categories=args.categories,
            markers=args.markers,
            verbose=args.verbose,
            coverage=not args.no_coverage,
            parallel=args.parallel,
            output_format=args.output_format
        )
    
    # Print summary
    if args.output_format == "text":
        print("\n" + "=" * 60)
        print("Test Execution Summary")
        print("=" * 60)
        print(f"Return Code: {results['return_code']}")
        print(f"Duration: {results['duration']:.2f} seconds")
        
        if 'summary' in results:
            summary = results['summary']
            print(f"Tests: {summary['total_tests']} total, "
                  f"{summary['passed']} passed, "
                  f"{summary['failed']} failed")
        
        if 'coverage' in results:
            coverage = results['coverage']
            print(f"Coverage: {coverage['total_coverage']:.1f}%")
    
    # Exit with appropriate code
    sys.exit(results['return_code'])


if __name__ == "__main__":
    main()
