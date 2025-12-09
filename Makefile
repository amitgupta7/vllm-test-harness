.PHONY: apply delete

deploy:
	kubectl apply -k jupyterML

delete:
	kubectl delete -k jupyterML

redeploy: delete deploy
	echo "redeploying."