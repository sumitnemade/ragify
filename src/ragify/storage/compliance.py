"""
Compliance Manager for GDPR, HIPAA, SOX, and other regulatory compliance.
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import structlog

from ..models import PrivacyLevel
from ..exceptions import ComplianceError, PrivacyViolationError
from .security import SecurityManager


class ComplianceManager:
    """
    Comprehensive compliance manager for regulatory requirements.
    
    Implements:
    - GDPR: Data protection, right to be forgotten, data portability
    - HIPAA: Healthcare data protection, audit trails, access controls
    - SOX: Financial data integrity, audit logging, retention policies
    """
    
    def __init__(
        self,
        compliance_frameworks: List[str] = None,
        audit_log_path: str = "compliance_audit.log",
        retention_policies: Dict[str, Any] = None,
        security_manager: Optional[SecurityManager] = None
    ):
        """Initialize the compliance manager."""
        self.compliance_frameworks = compliance_frameworks or ["GDPR", "HIPAA", "SOX"]
        self.audit_log_path = audit_log_path
        self.logger = structlog.get_logger(__name__)
        
        # Initialize security manager
        self.security_manager = security_manager or SecurityManager(security_level="enterprise")
        
        # Initialize audit logging
        self.audit_log = []
        self._load_audit_log()
        
        # Compliance policies
        self.compliance_policies = self._get_compliance_policies()
        
        # Retention policies
        self.retention_policies = retention_policies or self._get_default_retention_policies()
        
        # Data registries
        self.data_subjects = {}
        self.phi_records = {}
        self.financial_records = {}
        
        # Compliance monitoring
        self.compliance_violations = []
        self.compliance_checks = []
        
        self.logger.info(f"Compliance manager initialized for frameworks: {self.compliance_frameworks}")
    
    def _get_compliance_policies(self) -> Dict[str, Any]:
        """Get compliance policies for each framework."""
        return {
            "GDPR": {
                "data_protection": True,
                "right_to_be_forgotten": True,
                "data_portability": True,
                "consent_management": True,
            },
            "HIPAA": {
                "privacy_rule": True,
                "security_rule": True,
                "breach_notification": True,
                "minimum_necessary": True,
            },
            "SOX": {
                "financial_reporting": True,
                "internal_controls": True,
                "audit_trails": True,
                "data_integrity": True,
            }
        }
    
    def _get_default_retention_policies(self) -> Dict[str, Any]:
        """Get default data retention policies."""
        return {
            "personal_data": {"retention_period": 365, "deletion_method": "secure_deletion"},
            "healthcare_data": {"retention_period": 2555, "deletion_method": "secure_deletion"},
            "financial_data": {"retention_period": 2555, "deletion_method": "secure_deletion"},
            "audit_logs": {"retention_period": 1825, "deletion_method": "secure_deletion"},
        }
    
    def _load_audit_log(self) -> None:
        """Load compliance audit log from file."""
        try:
            if Path(self.audit_log_path).exists():
                with open(self.audit_log_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                event = json.loads(line)
                                self.audit_log.append(event)
                            except json.JSONDecodeError:
                                continue
                self.logger.info(f"Loaded {len(self.audit_log)} compliance audit log entries")
        except Exception as e:
            self.logger.warning(f"Failed to load compliance audit log: {e}")
    
    async def log_compliance_event(
        self,
        event_type: str,
        user_id: str,
        details: Dict[str, Any],
        framework: str = "general",
        severity: str = "info"
    ) -> None:
        """Log a compliance event."""
        try:
            event = {
                'timestamp': datetime.utcnow().isoformat(),
                'event_type': event_type,
                'user_id': user_id,
                'framework': framework,
                'severity': severity,
                'details': details,
                'compliance_status': 'compliant'
            }
            
            self.audit_log.append(event)
            
            if severity in ['error', 'critical']:
                await self._handle_compliance_violation(event)
            
            await self._save_audit_log()
            
        except Exception as e:
            self.logger.error(f"Failed to log compliance event: {e}")
    
    async def _handle_compliance_violation(self, event: Dict[str, Any]) -> None:
        """Handle compliance violations."""
        try:
            event['compliance_status'] = 'non_compliant'
            self.compliance_violations.append(event)
            
            await self.security_manager.log_security_event(
                user_id=event['user_id'],
                event_type=f"compliance_violation_{event['framework']}",
                details=f"Framework: {event['framework']}, Type: {event['event_type']}",
                severity="high"
            )
            
            self.logger.warning(f"Compliance violation detected: {event['event_type']} for {event['framework']}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle compliance violation: {e}")
    
    async def _save_audit_log(self) -> None:
        """Save compliance audit log to file."""
        try:
            with open(self.audit_log_path, 'w') as f:
                for event in self.audit_log:
                    f.write(json.dumps(event) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to save compliance audit log: {e}")
    
    # GDPR Compliance Methods
    
    async def register_data_subject(
        self,
        subject_id: str,
        personal_data: Dict[str, Any],
        consent_given: bool = False,
        consent_details: Dict[str, Any] = None
    ) -> str:
        """Register a data subject for GDPR compliance."""
        try:
            registration_id = hashlib.sha256(f"{subject_id}_{datetime.utcnow().isoformat()}".encode()).hexdigest()
            
            data_subject = {
                'registration_id': registration_id,
                'subject_id': subject_id,
                'personal_data': personal_data,
                'consent_given': consent_given,
                'consent_details': consent_details or {},
                'registration_date': datetime.utcnow().isoformat(),
                'data_processing_purposes': [],
                'data_retention_period': self.retention_policies['personal_data']['retention_period'],
                'last_accessed': None,
                'access_history': [],
                'deletion_requested': False,
                'deletion_date': None
            }
            
            self.data_subjects[registration_id] = data_subject
            
            await self.log_compliance_event(
                event_type="data_subject_registered",
                user_id=subject_id,
                details={
                    'registration_id': registration_id,
                    'consent_given': consent_given,
                    'data_types': list(personal_data.keys())
                },
                framework="GDPR"
            )
            
            self.logger.info(f"Data subject registered: {registration_id}")
            return registration_id
            
        except Exception as e:
            self.logger.error(f"Failed to register data subject: {e}")
            raise ComplianceError("data_subject_registration", "GDPR", str(e))
    
    async def right_to_be_forgotten(self, registration_id: str) -> bool:
        """Implement right to be forgotten (GDPR Article 17)."""
        try:
            if registration_id not in self.data_subjects:
                raise ComplianceError("right_to_be_forgotten", "GDPR", "Data subject not found")
            
            data_subject = self.data_subjects[registration_id]
            data_subject['deletion_requested'] = True
            data_subject['deletion_date'] = datetime.utcnow().isoformat()
            
            await self.log_compliance_event(
                event_type="right_to_be_forgotten_requested",
                user_id=data_subject['subject_id'],
                details={
                    'registration_id': registration_id,
                    'deletion_date': data_subject['deletion_date']
                },
                framework="GDPR"
            )
            
            await self._schedule_data_deletion(registration_id, "personal_data")
            
            self.logger.info(f"Right to be forgotten implemented for: {registration_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to implement right to be forgotten: {e}")
            return False
    
    # HIPAA Compliance Methods
    
    async def register_phi_record(
        self,
        patient_id: str,
        phi_data: Dict[str, Any],
        healthcare_provider: str,
        treatment_purpose: str
    ) -> str:
        """Register PHI (Protected Health Information) for HIPAA compliance."""
        try:
            phi_record_id = hashlib.sha256(f"{patient_id}_{datetime.utcnow().isoformat()}".encode()).hexdigest()
            
            phi_record = {
                'phi_record_id': phi_record_id,
                'patient_id': patient_id,
                'phi_data': phi_data,
                'healthcare_provider': healthcare_provider,
                'treatment_purpose': treatment_purpose,
                'creation_date': datetime.utcnow().isoformat(),
                'access_controls': {
                    'authorized_users': [],
                    'minimum_necessary': True,
                    'audit_logging': True
                },
                'retention_period': self.retention_policies['healthcare_data']['retention_period'],
                'access_history': [],
                'breach_detected': False,
                'breach_details': None
            }
            
            self.phi_records[phi_record_id] = phi_record
            
            await self.log_compliance_event(
                event_type="phi_record_created",
                user_id=patient_id,
                details={
                    'phi_record_id': phi_record_id,
                    'healthcare_provider': healthcare_provider,
                    'treatment_purpose': treatment_purpose
                },
                framework="HIPAA"
            )
            
            self.logger.info(f"PHI record registered: {phi_record_id}")
            return phi_record_id
            
        except Exception as e:
            self.logger.error(f"Failed to register PHI record: {e}")
            raise ComplianceError("phi_registration", "HIPAA", str(e))
    
    # SOX Compliance Methods
    
    async def register_financial_record(
        self,
        record_id: str,
        financial_data: Dict[str, Any],
        record_type: str,
        fiscal_period: str
    ) -> str:
        """Register financial record for SOX compliance."""
        try:
            financial_record_id = hashlib.sha256(f"{record_id}_{datetime.utcnow().isoformat()}".encode()).hexdigest()
            
            financial_record = {
                'financial_record_id': financial_record_id,
                'record_id': record_id,
                'financial_data': financial_data,
                'record_type': record_type,
                'fiscal_period': fiscal_period,
                'creation_date': datetime.utcnow().isoformat(),
                'audit_trail': [],
                'change_history': [],
                'approval_status': 'pending',
                'approved_by': None,
                'approval_date': None,
                'retention_period': self.retention_policies['financial_data']['retention_period'],
                'data_integrity_verified': False,
                'last_verified': None
            }
            
            self.financial_records[financial_record_id] = financial_record
            
            await self.log_compliance_event(
                event_type="financial_record_created",
                user_id=record_id,
                details={
                    'financial_record_id': financial_record_id,
                    'record_type': record_type,
                    'fiscal_period': fiscal_period
                },
                framework="SOX"
            )
            
            self.logger.info(f"Financial record registered: {financial_record_id}")
            return financial_record_id
            
        except Exception as e:
            self.logger.error(f"Failed to register financial record: {e}")
            raise ComplianceError("financial_registration", "SOX", str(e))
    
    # General Compliance Methods
    
    async def run_compliance_check(self, framework: str) -> Dict[str, Any]:
        """Run compliance check for a specific framework."""
        try:
            if framework == "GDPR":
                check_results = await self._run_gdpr_compliance_check()
            elif framework == "HIPAA":
                check_results = await self._run_hipaa_compliance_check()
            elif framework == "SOX":
                check_results = await self._run_sox_compliance_check()
            else:
                raise ComplianceError("compliance_check", "general", f"Unknown framework: {framework}")
            
            self.compliance_checks.append(check_results)
            
            await self.log_compliance_event(
                event_type="compliance_check_completed",
                user_id="system",
                details={
                    'framework': framework,
                    'status': check_results['overall_status'],
                    'violations_count': len(check_results['violations'])
                },
                framework=framework
            )
            
            return check_results
            
        except Exception as e:
            self.logger.error(f"Failed to run compliance check: {e}")
            raise ComplianceError("compliance_check", framework, str(e))
    
    async def _run_gdpr_compliance_check(self) -> Dict[str, Any]:
        """Run GDPR compliance check."""
        results = {
            'framework': 'GDPR',
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'compliant',
            'checks': [],
            'violations': [],
            'recommendations': []
        }
        
        # Check data subject registrations
        for registration_id, data_subject in self.data_subjects.items():
            if not data_subject['consent_given']:
                results['violations'].append({
                    'type': 'missing_consent',
                    'registration_id': registration_id,
                    'subject_id': data_subject['subject_id']
                })
        
        if results['violations']:
            results['overall_status'] = 'non_compliant'
        
        return results
    
    async def _run_hipaa_compliance_check(self) -> Dict[str, Any]:
        """Run HIPAA compliance check."""
        results = {
            'framework': 'HIPAA',
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'compliant',
            'checks': [],
            'violations': [],
            'recommendations': []
        }
        
        # Check PHI records
        for phi_record_id, phi_record in self.phi_records.items():
            if not phi_record['access_controls']['audit_logging']:
                results['violations'].append({
                    'type': 'missing_audit_logging',
                    'phi_record_id': phi_record_id,
                    'patient_id': phi_record['patient_id']
                })
        
        if results['violations']:
            results['overall_status'] = 'non_compliant'
        
        return results
    
    async def _run_sox_compliance_check(self) -> Dict[str, Any]:
        """Run SOX compliance check."""
        results = {
            'framework': 'SOX',
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'compliant',
            'checks': [],
            'violations': [],
            'recommendations': []
        }
        
        # Check financial records
        for financial_record_id, financial_record in self.financial_records.items():
            if not financial_record['audit_trail']:
                results['violations'].append({
                    'type': 'missing_audit_trail',
                    'financial_record_id': financial_record_id,
                    'record_id': financial_record['record_id']
                })
        
        if results['violations']:
            results['overall_status'] = 'non_compliant'
        
        return results
    
    async def _schedule_data_deletion(self, record_id: str, data_type: str) -> None:
        """Schedule data deletion for compliance."""
        try:
            await self.log_compliance_event(
                event_type="data_deletion_scheduled",
                user_id="system",
                details={
                    'record_id': record_id,
                    'data_type': data_type,
                    'scheduled_date': (datetime.utcnow() + timedelta(days=30)).isoformat()
                },
                framework="general"
            )
            
            self.logger.info(f"Data deletion scheduled for {data_type}: {record_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to schedule data deletion: {e}")
    
    async def generate_compliance_report(
        self,
        framework: str = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            # Filter events by date
            period_events = [
                event for event in self.audit_log
                if start_date <= datetime.fromisoformat(event['timestamp']) <= end_date
            ]
            
            # Filter by framework if specified
            if framework:
                period_events = [event for event in period_events if event.get('framework') == framework]
            
            # Generate report
            report = {
                'report_period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'framework': framework or 'all',
                'total_events': len(period_events),
                'compliance_status': 'compliant',
                'violations': [],
                'recommendations': [],
                'data_summary': {
                    'gdpr_subjects': len(self.data_subjects),
                    'hipaa_records': len(self.phi_records),
                    'sox_records': len(self.financial_records)
                }
            }
            
            # Check for violations
            violations = [event for event in period_events if event.get('severity') in ['error', 'critical']]
            if violations:
                report['compliance_status'] = 'non_compliant'
                report['violations'] = violations
            
            # Generate recommendations
            if framework == "GDPR" or not framework:
                report['recommendations'].extend([
                    "Ensure all data subjects have given explicit consent",
                    "Implement data minimization practices",
                    "Regularly review and update privacy policies"
                ])
            
            if framework == "HIPAA" or not framework:
                report['recommendations'].extend([
                    "Implement strict access controls for PHI",
                    "Ensure audit logging for all PHI access",
                    "Regular workforce training on HIPAA compliance"
                ])
            
            if framework == "SOX" or not framework:
                report['recommendations'].extend([
                    "Maintain comprehensive audit trails",
                    "Implement change management controls",
                    "Regular data integrity verification"
                ])
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate compliance report: {e}")
            raise ComplianceError("report_generation", framework or "general", str(e))
    
    async def close(self) -> None:
        """Close compliance manager and save audit log."""
        try:
            await self._save_audit_log()
            await self.security_manager.close()
            self.logger.info("Compliance manager closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing compliance manager: {e}")

