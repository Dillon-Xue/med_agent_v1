import contextvars

doctor_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("doctor_id", default="")
tenant_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("tenant_id", default="default")
